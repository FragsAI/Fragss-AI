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
    """ Dynamically adjust frame sampling interval based on video length. """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Video duration in seconds
    cap.release()

    # Adjust sampling based on video duration
    if duration <= 3600:  # 1 hour
        return 10  # Sample every 10th frame
    elif duration <= 18000:  # 5 hours
        return 15  # Sample every 15th frame
    else:
        return 20  # Sample every 20th frame

def determine_chunk_size():
    """ Adjust chunk size dynamically based on available memory. """
    available_memory = psutil.virtual_memory().available  # Get available RAM in bytes

    if available_memory < 4 * 1024**3:  # Less than 4GB RAM
        return 500  # Small chunks
    elif available_memory < 8 * 1024**3:  # Less than 8GB RAM
        return 1000
    elif available_memory < 16 * 1024**3:  # Less than 16GB RAM
        return 2000
    else:
        return 5000  # Large chunks for high-memory systems

# ---------------------- Frame Extraction and Processing ----------------------

def frame_generator(video_path, sample_interval, downscale_factor):
    """ Generator function to extract frames at a given interval efficiently. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // downscale_factor
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // downscale_factor
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        yield frame

        frame_index += sample_interval  # Skip frames at the given interval

    cap.release()
    cv2.destroyAllWindows()

def extract_frames(video_path, output_folder, sample_interval, downscale_factor, batch_size):
    """ Extract frames from video and save them in efficient batches. """
    os.makedirs(output_folder, exist_ok=True)

    batch = []
    batch_counter = 0
    progress_bar = tqdm(desc="Extracting frames", unit=" frame")

    for frame in frame_generator(video_path, sample_interval, downscale_factor):
        batch.append(frame)

        if len(batch) >= batch_size:
            np.save(os.path.join(output_folder, f"batch_{batch_counter:04d}.npy"), np.array(batch, dtype=np.uint8))
            logging.info(f" Saved batch {batch_counter} with {len(batch)} frames.")
            batch_counter += 1
            batch.clear()
            gc.collect()

        progress_bar.update(1)

    if batch:
        np.save(os.path.join(output_folder, f"batch_{batch_counter:04d}.npy"), np.array(batch, dtype=np.uint8))
        logging.info(f" Saved final batch {batch_counter} with {len(batch)} frames.")

    progress_bar.close()
    logging.info("Frame extraction completed successfully.")



# ---------------------- Convert Images to Grayscale ----------------------

def convert_to_grayscale(frame):
    """Convert a frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ---------------------- Frame Augmentation ----------------------

def apply_augmentations(frame):
    """Apply augmentation techniques to a frame."""
    augmentations = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HorizontalFlip(p=0.5),
        GaussianBlur(blur_limit=3, p=0.3)
    ])
    augmented_frame = augmentations(image=frame)['image']
    if np.random.random() < 0.3:
        augmented_frame = random_noise(augmented_frame, mode='gaussian', var=0.01)
        augmented_frame = (augmented_frame * 255).astype(np.uint8)
    return augmented_frame

# ---------------------- Process Frames from Batches ----------------------

# def process_frames(batch_folder, output_file, resize_dim, augment=True):
#     """Process frames by resizing, augmenting (or converting to grayscale), normalizing, and saving."""
#     batch_files = sorted([f for f in os.listdir(batch_folder) if f.endswith(".npy")])
#     logging.info(f"Processing {len(batch_files)} batch files.")

#     processed_frames = []
#     with tqdm(total=len(batch_files), desc="Processing frames") as progress:
#         for batch_file in batch_files:
#             batch_path = os.path.join(batch_folder, batch_file)
#             frames = np.load(batch_path)

#             with ThreadPoolExecutor(max_workers=4) as executor:
#                 frames = list(executor.map(apply_augmentations, frames)) if augment else list(executor.map(convert_to_grayscale, frames))

#             frames = [cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA) for frame in frames]
#             frames = np.array(frames, dtype=np.float32) / 255.0
#             processed_frames.extend(frames)

#             progress.update(1)

#             if len(processed_frames) >= 1000:
#                 temp_output = output_file.replace(".npy", f"_{progress.n}.npy")
#                 np.save(temp_output, np.array(processed_frames, dtype=np.float32))
#                 processed_frames.clear()
#                 gc.collect()

#     if processed_frames:
#         np.save(output_file, np.array(processed_frames, dtype=np.float32))
#         logging.info(f"Processed frames saved to {output_file}")

def process_frames(batch_folder, output_file, resize_dim, augment=True):
    """Process frames by resizing, augmenting (or converting to grayscale), normalizing, and saving."""
    gray_scale_npy_file_name = 'gray_frame.npy' # .npy file to save a array of gray scaled frame
    final_gray_scaled_frames_dir = os.path.join(output_file, gray_scale_npy_file_name)

    processed_frames = []
    
    # orb = cv2.ORB_create()  # Initialize ORB
    
    batch_files = sorted([f for f in os.listdir(batch_folder) if f.endswith(".npy")])
    logging.info(f"Processing {len(batch_files)} batch files.")
    
    with tqdm(total=len(batch_files), desc="Processing frames") as progress:
        for i, batch_file in enumerate(batch_files):
            batch_path = os.path.join(batch_folder, batch_file)
            frames = np.load(batch_path)

            with ThreadPoolExecutor(max_workers=4) as executor:
                frames = list(executor.map(apply_augmentations, frames)) if augment else list(executor.map(convert_to_grayscale, frames))

            frames = [cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA) for frame in frames]
            
            # Convert back to uint8 for ORB processing
            frames = [(frame * 255).astype(np.uint8) for frame in frames]

            # for frame in frames:
            #     keypoints, descriptors = orb.detectAndCompute(frame, None)
            #     if descriptors is None:
            #         logging.warning("No keypoints detected in a frame.")

            processed_frames.extend(frames)

            progress.update(1)

            if len(processed_frames) >= 1000:
                temp_output = final_gray_scaled_frames_dir.replace(".npy", f"_{i}.npy")
                np.save(temp_output, np.array(processed_frames, dtype=np.uint8))  # Save as uint8
                processed_frames.clear()
                gc.collect()
                
    # logging.info(f"Processed frames saved to {output_file}")
    if processed_frames:
        np.save(final_gray_scaled_frames_dir.replace(".npy", f"_{len(batch_files)-1}.npy"), np.array(processed_frames, dtype=np.uint8))  # Ensure final file is uint8
        logging.info(f" Processed frames saved to {output_file}")
                
    # ---------------------- Main Function ----------------------


# def main():
#     """
#     Main function to extract, preprocess, and augment video frames.
#     """
#     video_path = r"C:\Users\chimm\Frags AI\FragsAIVideosAndAudios\Fortnite Battle Royale.mp4"
#     output_dir = r"C:\Users\chimm\Frags AI\FragsAIVideosAndAudios\FramesGenerated"
#     gray_scaled_frames_dir = r"C:\Users\chimm\Frags AI\FragsAIVideosAndAudios\GrayFrames\gray_scaled_frames_batch.npy"
    
#     sample_interval = adjust_sample_interval(video_path)
#     print(f"No.of frames to skip: {sample_interval}")
#     chunk_size = determine_chunk_size()
#     print(f"Chunk size to assign as batch size: {chunk_size}\n")

#     print('Frames extraction begins:')
#     extract_frames(video_path, output_dir, sample_interval, downscale_factor=2, batch_size=chunk_size)
#     print('\nProcessing and converting frames to gray scale:')
#     process_frames(output_dir, gray_scaled_frames_dir, resize_dim=(224, 224), augment=False)

# start = timer()
# print('Process begins\n')
# if __name__ == "__main__":
#     main()

# end = timer()
# duration = end-start
# print(f"End! Time taken {round(duration/60)}m {round(duration%60)}s")