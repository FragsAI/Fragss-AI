import cv2
import numpy as np
import os
import logging
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from skimage.util import random_noise


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video and saves them as JPEG files.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save the extracted frames.
        frame_rate (int): Interval of frames to extract (e.g., 1 for every frame, 5 for every 5th frame).
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return
    
        frame_paths = []
        frame_count = 0
        extracted_frames = 0
        missed_frames = 0
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=tot_frames//frame_rate, desc="Extracting frames")
        logging.info(f"Skipping every {frame_rate} frames")
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
                    frame_path = os.path.join(output_folder, f"frame_{extracted_frames:06d}.jpg")
                    if cv2.imwrite(frame_path, frame):
                        extracted_frames += 1
                        frame_paths.append(frame_path)
                        progress_bar.update(1)
                    else:
                        logging.warning(f"Failed to write frame: {frame_path}")
                        missed_frames += 1

            frame_count += 1

        cap.release()
        progress_bar.close()

        cv2.destroyAllWindows()
        logging.info(f"Extracted {extracted_frames} frames. Missed {missed_frames} frames.")

        return frame_paths
    
    except Exception as e:
        logging.error(f"An error occurred during frame extraction: {e}")

def augment_frame(frame):
    """
    Apply random transformations to augment the frame.

    Args:
        frame (numpy.ndarray): Frame to augment.
    
    Returns:
        numpy.ndarray: Augmented frame.
    """
    if random.choice([True, False]):
        frame = cv2.flip(frame, 1)  # Horizontal flip
    
    if random.choice([True, False]):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Gaussian blur
    
    if random.choice([True, False]):
        frame = random_noise(frame, mode='speckle')  # Add speckle noise
        frame = np.array(255 * frame, dtype='uint8')
    
    return frame

def preprocess_frames(input_folder, output_file, resize_dim=(224, 224), augment=False):
    """
    Preprocesses extracted frames by resizing and normalizing them, and saves as a numpy array.

    Args:
        input_folder (str): Directory containing the extracted frames.
        output_file (str): Path to save the preprocessed frames numpy array.
        resize_dim (tuple): Dimensions to resize frames to.
        augment (bool): Whether to apply augmentation to the frames.
    """
    try:
        frames = []
        frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
        total_frames = len(frame_files)
        logging.info(f"Total frames to preprocess: {total_frames}")

        with tqdm(total=total_frames, desc="Preprocessing frames") as pbar:
            for file_name in frame_files:
                frame_path = os.path.join(input_folder, file_name)
                frame = cv2.imread(frame_path)
                if frame is None:
                    logging.warning(f"Could not read frame: {frame_path}")
                    continue
                frame = cv2.resize(frame, resize_dim)
                frame = frame.astype(np.float32) / 255.0
                if augment:
                    frame = augment_frame(frame)
                frames.append(frame)
                pbar.update(1)
        
        frames = np.array(frames)
        np.save(output_file, frames)
        logging.info(f"Preprocessed frames saved to {output_file}")
    
    except Exception as e:
        logging.error(f"An error occurred during frame preprocessing: {e}")


def main():
    video_path = 'your video path here'
    output_dir = 'extracted_frames'
    frame_rate = 5

    frame_paths = extract_frames(video_path, output_dir, frame_rate=frame_rate)
    print(f"Extracted and processed {len(frame_paths)} frames.")

    augmented_output_dir =  'augmented'
    os.makedirs(augmented_output_dir, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda frame_path: preprocess_frames(frame_path, augmented_output_dir), frame_paths), total=len(frame_paths), desc="Augmenting frames"))

if __name__ == "__main__":
    main()
