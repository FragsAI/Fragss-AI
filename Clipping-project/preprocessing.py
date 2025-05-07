import cv2
import numpy as np
import os
import logging
import gc
import psutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, GaussianBlur
from skimage.util import random_noise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Memory and Sampling Configuration ----------------------

def adjust_sample_interval(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Video duration in seconds
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

# ---------------------- Frame Extraction ----------------------

def frame_generator(video_path, sample_interval, downscale_factor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // downscale_factor
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // downscale_factor
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    while frame_index < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % sample_interval == 0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            yield frame
        frame_index += 1

    cap.release()

# ---------------------- Frame Saving ----------------------

def extract_frames(video_path, output_folder, sample_interval, downscale_factor, batch_size):
    os.makedirs(output_folder, exist_ok=True)

    batch = []
    batch_counter = 0
    progress_bar = tqdm(desc="Extracting frames", unit=" frame")

    for frame in frame_generator(video_path, sample_interval, downscale_factor):
        batch.append(frame)

        if len(batch) >= batch_size:
            np.save(os.path.join(output_folder, f"batch_{batch_counter:04d}.npy"), np.array(batch, dtype=np.uint8))
            logging.info(f"Saved batch {batch_counter} with {len(batch)} frames.")
            batch_counter += 1
            batch.clear()
            gc.collect()

        progress_bar.update(1)

    if batch:
        np.save(os.path.join(output_folder, f"batch_{batch_counter:04d}.npy"), np.array(batch, dtype=np.uint8))
        logging.info(f"Saved final batch {batch_counter} with {len(batch)} frames.")

    progress_bar.close()
    logging.info("Frame extraction completed successfully.")

# ---------------------- Frame Augmentation ----------------------

def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_augmentations(frame):
    augmentations = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HorizontalFlip(p=0.5),
        GaussianBlur(blur_limit=3, p=0.3)
    ])
    frame = augmentations(image=frame)['image']
    if np.random.random() < 0.3:
        frame = random_noise(frame, mode='gaussian', var=0.01)
        frame = (frame * 255).astype(np.uint8)
    return frame

# ---------------------- Processing Frames ----------------------

def process_frames(batch_folder, output_folder, resize_dim=(224, 224), augment=True):
    os.makedirs(output_folder, exist_ok=True)

    batch_files = sorted([f for f in os.listdir(batch_folder) if f.endswith(".npy")])
    logging.info(f"Processing {len(batch_files)} batch files.")

    processed_frames = []
    with tqdm(total=len(batch_files), desc="Processing frames") as progress:
        for i, batch_file in enumerate(batch_files):
            batch_path = os.path.join(batch_folder, batch_file)
            frames = np.load(batch_path)

            with ThreadPoolExecutor(max_workers=4) as executor:
                frames = list(executor.map(apply_augmentations, frames)) if augment else list(executor.map(convert_to_grayscale, frames))

            frames = [cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA) for frame in frames]
            frames = [(frame * 255).astype(np.uint8) for frame in frames]

            processed_frames.extend(frames)

            if len(processed_frames) >= 1000:
                np.save(os.path.join(output_folder, f"processed_{i:04d}.npy"), np.array(processed_frames, dtype=np.uint8))
                processed_frames.clear()
                gc.collect()

            progress.update(1)

    if processed_frames:
        np.save(os.path.join(output_folder, f"processed_final.npy"), np.array(processed_frames, dtype=np.uint8))
        logging.info("Saved final processed frames.")

    logging.info("Processing complete.")
