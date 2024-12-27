import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import concurrent.futures
from collections import deque
from scipy.stats import norm
import os
from shot_sift import extract_frames_multithreaded

class FrameExtractor:
    def __init__(self, video_path, buffer_size=30):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.cap = None
        self.total_frames = 0
        self.initialize_video()

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def extract_frame_range(self, start_frame, num_frames):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def extract_frames_parallel(self, num_threads=4):
        frames_per_thread = self.total_frames // num_threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_frames = []
            for i in range(num_threads):
                start_frame = i * frames_per_thread
                num_frames = frames_per_thread if i < num_threads - 1 else self.total_frames - start_frame
                future = executor.submit(self.extract_frame_range, start_frame, num_frames)
                future_frames.append(future)
            
            all_frames = []
            for future in concurrent.futures.as_completed(future_frames):
                all_frames.extend(future.result())
        return all_frames

    def __del__(self):
        if self.cap:
            self.cap.release()

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def calculate_frame_difference(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    return np.mean(diff)

def extract_orb_features(frame, n_features=500):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors

def calculate_feature_similarity(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0.0
    
    # Use Hamming distance for ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    if len(matches) == 0:
        return 0.0
        
    avg_distance = sum(m.distance for m in matches) / len(matches)
    similarity = 1.0 / (1.0 + avg_distance)
    return similarity

def calculate_dynamic_threshold(differences, window_size=30):
    if len(differences) < window_size:
        return np.mean(differences) + 2 * np.std(differences)
    
    recent_diffs = differences[-window_size:]
    mu = np.mean(recent_diffs)
    sigma = np.std(recent_diffs)
    threshold = mu + 2 * sigma  # 2 sigma for 95% confidence
    return threshold

def detect_shot_boundaries(video_path, method='sift', diff_threshold=50, match_threshold=0.7, num_threads=4, frame_skip=1, min_shot_length=15):
    # Check if file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Try to open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    cap.release()
    
    frames = extract_frames_multithreaded(video_path, num_threads=num_threads, frame_skip=frame_skip)
    
    if not frames:
        print("Error: No frames extracted from video")
        return []
    
    shot_boundaries = [0]  # First frame is always a boundary
    frame_differences = []
    feature_similarities = []
    
    prev_frame = preprocess_frame(frames[0])
    prev_descriptors = None
    
    # Process frames
    for i in range(1, len(frames)):
        curr_frame = preprocess_frame(frames[i])
        
        # Calculate frame difference
        diff = calculate_frame_difference(prev_frame, curr_frame)
        frame_differences.append(diff)
        
        # Extract ORB features and calculate similarity
        _, curr_descriptors = extract_orb_features(curr_frame)
        if prev_descriptors is not None:
            similarity = calculate_feature_similarity(prev_descriptors, curr_descriptors)
            feature_similarities.append(similarity)
        
        # Calculate dynamic thresholds
        if len(frame_differences) >= 3:
            diff_threshold = calculate_dynamic_threshold(frame_differences)
            sim_threshold = calculate_dynamic_threshold(feature_similarities) if feature_similarities else 0.5
            
            # Combine frame difference and feature similarity for shot boundary detection
            is_boundary = (diff > diff_threshold and 
                        (len(feature_similarities) == 0 or feature_similarities[-1] < sim_threshold) and
                        (i - shot_boundaries[-1]) >= min_shot_length)
            
            if is_boundary:
                shot_boundaries.append(i)
        
        prev_frame = curr_frame
        prev_descriptors = curr_descriptors
    
    # Convert frame indices to timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    timestamps = [frame_idx / fps * 1000 for frame_idx in shot_boundaries]
    return timestamps

if __name__ == "__main__":
    video_path = "/Users/rnzgrd/Downloads/SOLO_VS_SQUAD_34_KILLS_FULL_GAMEPLAY_CALL_OF_DUTY_MOBILE_BATTLE_ROYALE.mp4"
    shot_boundaries = detect_shot_boundaries(video_path)
    print(f"Shot boundaries detected at (ms): {shot_boundaries}")
