# preprocessing.py

import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
from skimage.util import random_noise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video and saves them as JPEG files at a specified frame interval.
    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save the extracted frames.
        frame_rate (int): Interval of frames to extract (1 for every frame, 5 for every 5th frame, etc.).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return frame_paths
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames // frame_rate, desc="Extracting frames") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                pbar.update(1)
            frame_count += 1

    cap.release()
    return frame_paths

def augment_frame(frame):
    """
    Apply random augmentations to the frame for data variability.
    Args:
        frame (numpy.ndarray): Frame to augment.
    Returns:
        numpy.ndarray: Augmented frame.
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        frame = cv2.flip(frame, 1)
    
    # Random Gaussian blur
    if np.random.rand() > 0.5:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Random speckle noise
    if np.random.rand() > 0.5:
        frame = random_noise(frame, mode='speckle')
        frame = np.array(255 * frame, dtype='uint8')
    
    return frame

def preprocess_frames(input_folder, output_file, resize_dim=(224, 224), augment=False):
    """
    Resize and normalize frames, applying augmentation if specified, and save as a numpy array.
    Args:
        input_folder (str): Directory containing the extracted frames.
        output_file (str): Path to save the preprocessed frames numpy array.
        resize_dim (tuple): Dimensions to resize frames to.
        augment (bool): Whether to apply augmentation to the frames.
    """
    frames = []
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
    for file_name in tqdm(frame_files, desc="Preprocessing frames"):
        frame_path = os.path.join(input_folder, file_name)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, resize_dim)
            frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
            if augment:
                frame = augment_frame(frame)
            frames.append(frame)
    np.save(output_file, np.array(frames))
    logging.info(f"Preprocessed frames saved to {output_file}")
    return output_file

def preprocess_video_pipeline(video_path, output_dir='processed_frames', frame_rate=5, resize_dim=(224, 224), augment=True):
    """
    Full preprocessing pipeline: frame extraction, resizing, normalization, and augmentation.
    Args:
        video_path (str): Path to the uploaded video file.
        output_dir (str): Directory to save processed frames.
        frame_rate (int): Interval of frames to extract.
        resize_dim (tuple): Dimensions to resize frames to.
        augment (bool): Whether to apply augmentation.
    """
    logging.info(f"Starting preprocessing pipeline for {video_path}")
    
    # Step 1: Extract frames from the video
    frame_paths = extract_frames(video_path, output_dir, frame_rate=frame_rate)
    logging.info(f"Extracted {len(frame_paths)} frames")

    # Step 2: Preprocess and optionally augment frames
    preprocessed_output = os.path.join(output_dir, 'preprocessed_frames.npy')
    preprocess_frames(output_dir, preprocessed_output, resize_dim=resize_dim, augment=augment)
    return preprocessed_output
