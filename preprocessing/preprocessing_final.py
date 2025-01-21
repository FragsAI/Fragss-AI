import cv2
import numpy as np
import os
import logging
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, HorizontalFlip, GaussianBlur
from skimage.util import random_noise


# Set up logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames(video_path, output_folder, frame_rate=1, downscale_factor=2, batch_size=100):
    """
    Extracts frames from a video, processes them in batches, and saves them in batch files to reduce disk I/O.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save the extracted frames.
        frame_rate (int): Interval of frames to extract (e.g., every 5th frame).
        downscale_factor (int): Factor to downscale resolution.
        batch_size (int): Number of frames to process in a batch.
    """
    try:
        # Ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return

        # Get video properties
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Video properties: {tot_frames} frames, {fps:.2f} FPS, {original_width}x{original_height} resolution.")

        # Calculate downscaled dimensions
        width = original_width // downscale_factor
        height = original_height // downscale_factor
        logging.info(f"Frames will be downscaled to {width}x{height}")

        # Progress bar
        progress_bar = tqdm(total=tot_frames // frame_rate, desc="Extracting frames")
        frame_batch = []  # Temporary storage for the batch
        batch_counter = 0  # Tracks batch numbers

        def save_batch(batch, batch_index):
            """Helper function to save a batch of frames to disk."""
            batch_path = os.path.join(output_folder, f"batch_{batch_index:04d}.npy")
            np.save(batch_path, np.array(batch))
            logging.info(f"Batch {batch_index} saved with {len(batch)} frames.")

        while True:
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:  # No more frames
                break

            # Process only frames at the specified interval
            if frame_index % frame_rate == 0:
                # Downscale the frame
                if downscale_factor > 1:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Add frame to the batch
                frame_batch.append(frame)

                # Save batch when it's full
                if len(frame_batch) >= batch_size:
                    save_batch(frame_batch, batch_counter)
                    batch_counter += 1
                    frame_batch = []  # Clear the batch
                    progress_bar.update(batch_size)

        # Save any remaining frames in the last batch
        if frame_batch:
            save_batch(frame_batch, batch_counter)
            progress_bar.update(len(frame_batch))

        # Clean up
        cap.release()
        progress_bar.close()
        cv2.destroyAllWindows()

        logging.info("Frame extraction completed successfully!")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


def augment_frame(frame):
    """
    Apply augmentations to a frame using Albumentations.

    Args:
        frame (numpy.ndarray): Frame to augment.

    Returns:
        numpy.ndarray: Augmented frame.
    """
    transform = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness/contrast adjustment
        RandomCrop(height=200, width=200, p=0.5),  # Random cropping
        HorizontalFlip(p=0.5),  # Horizontal flip
        GaussianBlur(blur_limit=3, p=0.3)  # Gaussian blur
    ])
    augmented = transform(image=frame)
    return augmented['image']


def preprocess_frames_from_batches(batch_folder, output_file, resize_dim=(224, 224), augment=True):
    """
    Preprocess extracted frames from .npy batches by resizing, augmenting, and saving as a single NumPy array.

    Args:
        batch_folder (str): Directory containing the .npy batch files.
        output_file (str): Path to save the preprocessed frames.
        resize_dim (tuple): Dimensions to resize frames to.
        augment (bool): Whether to apply augmentations to the frames.
    """
    try:
        frames = []
        batch_files = sorted([f for f in os.listdir(batch_folder) if f.endswith(".npy")])
        total_batches = len(batch_files)
        logging.info(f"Total batch files to preprocess: {total_batches}")

        with tqdm(total=total_batches, desc="Preprocessing batches") as pbar:
            for batch_file in batch_files:
                batch_path = os.path.join(batch_folder, batch_file)
                batch_frames = np.load(batch_path)

                for frame in batch_frames:
                    # Apply augmentation if enabled
                    if augment:
                        frame = augment_frame(frame)

                    # Resize frame to ensure consistent dimensions
                    frame = cv2.resize(frame, resize_dim)

                    # Normalize and add to the list
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)

                pbar.update(1)

        # Convert to NumPy array
        frames = np.array(frames)
        logging.info(f"Frames shape after preprocessing: {frames.shape}")

        # Save preprocessed frames
        np.save(output_file, frames)
        logging.info(f"Preprocessed frames saved to {output_file}")

    except Exception as e:
        logging.error(f"An error occurred during batch preprocessing: {e}")


def main():
    """
    Main function to extract, preprocess, and augment video frames.
    """
    video_path = 'your_video.mp4'  # Path to the input video
    output_dir = 'extracted_frames'  # Directory to save extracted frames
    augmented_output_file = 'augmented_frames.npy'  # Path to save augmented frames
    frame_rate = 5  # Extract every 5th frame

    # Step 1: Extract frames with batch saving
    logging.info("Step 1: Extracting frames...")
    extract_frames(video_path, output_dir, frame_rate=frame_rate)

    # Step 2: Preprocess frames (resize, augment, normalize)
    logging.info("Step 2: Preprocessing frames...")
    preprocess_frames_from_batches(output_dir, augmented_output_file, resize_dim=(224, 224), augment=True)



if __name__ == "__main__":
    main()
