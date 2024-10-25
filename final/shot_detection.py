import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("shot_boundary_detection.log"), logging.StreamHandler()])

# Extract SIFT features from a frame
def extract_sift_features(frame):
    sift = cv2.SIFT_create()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    return keypoints, descriptors

# Multithreaded frame extraction
def extract_frames_multithreaded(video_path, num_threads=4, frame_skip=1):
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

# Shot boundary detection using SIFT-based feature matching
def detect_shot_boundaries(video_path, method='sift', diff_threshold=50, match_threshold=0.7, num_threads=4, frame_skip=1):
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

# Build Temporal Convolutional Network (TCN) model for event classification
def build_tcn(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

# Main function to run shot boundary detection
def main():
    video_path = "input_video.mp4"
    method = 'sift'  # Can be 'diff' for frame differencing or 'sift' for SIFT-based detection
    frame_skip = 2  # Adjust this value based on your video frame rate and length
    num_threads = 4

    logging.info("Starting initial shot boundary detection")
    initial_boundaries = detect_shot_boundaries(video_path, method=method, frame_skip=frame_skip, num_threads=num_threads)

    logging.info("Starting shot boundary refinement")
    refined_boundaries = refine_shot_boundaries(video_path, initial_boundaries, num_threads=num_threads, frame_skip=frame_skip)

    print("Detected shot boundaries at frames:", refined_boundaries)

if __name__ == "__main__":
    main()
