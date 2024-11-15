# shot_detection.py

import os
import cv2
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Extract SIFT features from a frame
def extract_sift_features(frame):
    """
    Extract SIFT features from a frame.
    Args:
        frame (numpy.ndarray): Video frame.
    Returns:
        tuple: Keypoints and descriptors.
    """
    sift = cv2.SIFT_create()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    return keypoints, descriptors

# Multithreaded frame extraction
def extract_frames_multithreaded(video_path, num_threads=4, frame_skip=1):
    """
    Extract frames from a video using multithreading.
    Args:
        video_path (str): Path to the video file.
        num_threads (int): Number of threads to use.
        frame_skip (int): Frame interval to skip.
    Returns:
        dict: Extracted frames with indices as keys.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def process_frame(frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            return frame_index, frame
        return frame_index, None

    frame_indices = range(0, total_frames, frame_skip)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_frame, i) for i in frame_indices]
        frames = {i: f.result()[1] for i, f in zip(frame_indices, futures) if f.result()[1] is not None}

    cap.release()
    return frames

# Shot boundary detection using SIFT or frame differencing
def detect_shot_boundaries(video_path, method='sift', diff_threshold=50, match_threshold=0.7, num_threads=4, frame_skip=1):
    """
    Detects shot boundaries in a video.
    Args:
        video_path (str): Path to the video file.
        method (str): Method to use ('sift' or 'diff').
        diff_threshold (int): Threshold for frame difference method.
        match_threshold (float): Match threshold for SIFT method.
        num_threads (int): Number of threads for frame extraction.
        frame_skip (int): Frame interval to skip.
    Returns:
        list: Frame indices of detected shot boundaries.
    """
    frames = extract_frames_multithreaded(video_path, num_threads=num_threads, frame_skip=frame_skip)
    if method == 'sift':
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    prev_frame = None
    prev_des = None
    shot_boundaries = []

    logging.info(f"Total frames in video: {len(frames)}")

    with tqdm(total=len(frames), desc="Detecting shot boundaries") as pbar:
        for frame_index in sorted(frames.keys()):
            gray = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY)

            if method == 'diff':
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    non_zero_count = np.count_nonzero(diff)
                    if non_zero_count > diff_threshold:
                        shot_boundaries.append(frame_index)
                prev_frame = gray

            elif method == 'sift':
                kp, des = sift.detectAndCompute(gray, None)
                if prev_des is not None:
                    matches = bf.match(prev_des, des)
                    distances = [m.distance for m in matches]
                    if len(distances) > 0:
                        avg_distance = sum(distances) / len(distances)
                        if avg_distance > match_threshold:
                            shot_boundaries.append(frame_index)
                prev_des = des

            pbar.update(1)

    logging.info(f"Detected {len(shot_boundaries)} shot boundaries.")
    return shot_boundaries

# Refine shot boundaries using further analysis
def refine_shot_boundaries(video_path, initial_boundaries, threshold=0.7, num_threads=4, frame_skip=1):
    """
    Refines initial shot boundaries using feature matching.
    Args:
        video_path (str): Path to the video file.
        initial_boundaries (list): Initial shot boundaries.
        threshold (float): Refinement threshold.
        num_threads (int): Number of threads.
        frame_skip (int): Frame interval to skip.
    Returns:
        list: Refined shot boundaries.
    """
    frames = extract_frames_multithreaded(video_path, num_threads=num_threads, frame_skip=frame_skip)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    refined_boundaries = []
    prev_des = None

    for frame_index in sorted(frames.keys()):
        gray = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if prev_des is not None:
            matches = bf.match(prev_des, des)
            distances = [m.distance for m in matches]
            if len(distances) > 0:
                avg_distance = sum(distances) / len(distances)
                if avg_distance > threshold and frame_index in initial_boundaries:
                    refined_boundaries.append(frame_index)
        prev_des = des

    return refined_boundaries
