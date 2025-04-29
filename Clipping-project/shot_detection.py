import cv2
import numpy as np
import os
import logging
import gc
import psutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Frame Extraction ----------------------

def adjust_sample_interval(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    if duration <= 3600:
        return 10
    elif duration <= 18000:
        return 15
    else:
        return 20

def determine_chunk_size():
    available_memory = psutil.virtual_memory().available

    if available_memory < 4 * 1024**3:
        return 500
    elif available_memory < 8 * 1024**3:
        return 1000
    elif available_memory < 16 * 1024**3:
        return 2000
    else:
        return 5000

def process_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_index, gray
    return frame_index, None

def extract_frames_multithreaded(video_path, num_threads=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = adjust_sample_interval(video_path)
    frame_indices = list(range(0, total_frames, frame_skip))
    cap.release()

    frames = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(lambda idx: process_frame(video_path, idx), frame_indices), total=len(frame_indices), desc="Extracting frames"))
        frames = [gray for idx, gray in results if gray is not None]

    logging.info(f"Extracted {len(frames)} frames.")
    return frames

# ---------------------- Shot Detection ----------------------

def detect_shot_boundaries(frames, method='sift', diff_threshold=50, match_threshold=0.7):
    if method == 'sift':
        feature_detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == 'orb':
        feature_detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        feature_detector = None

    prev_des = None
    shot_boundaries = []

    logging.info(f"Detecting shot boundaries using {method.upper()} method...")

    for i in tqdm(range(len(frames)), desc="Shot detection"):
        frame = frames[i]

        if method == 'diff':
            if i > 0:
                diff = cv2.absdiff(frames[i-1], frame)
                non_zero_count = np.count_nonzero(diff)
                if non_zero_count > diff_threshold:
                    shot_boundaries.append(i)

        else:
            kp, des = feature_detector.detectAndCompute(frame, None)
            if prev_des is not None and des is not None:
                matches = matcher.match(prev_des, des)
                distances = [m.distance for m in matches]
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    if avg_distance > match_threshold:
                        shot_boundaries.append(i)
            prev_des = des

    logging.info(f"Detected {len(shot_boundaries)} shot boundaries.")
    return shot_boundaries

# Example usage
if __name__ == "__main__":
    video_path = "input_video.mp4"
    frames = extract_frames_multithreaded(video_path, num_threads=4)
    shot_boundaries = detect_shot_boundaries(frames, method='sift')
    print("Shot boundaries at frames:", shot_boundaries)
