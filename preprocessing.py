import cv2
import numpy as np
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_frames(video_path, output_dir, resize_dim=(224, 224), normalize=False):
    """
    Extract frames from a video, resize them, and normalize pixel values.

    Parameters:
    - video_path: Path to the video file.
    - output_dir: Directory where extracted frames will be saved.
    - resize_dim: Tuple specifying the size to resize frames to (width, height).
    - normalize: Boolean indicating whether to normalize pixel values to the range [0, 1].

    Returns:
    - frame_paths: List of paths to the extracted frame images.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    frame_paths = []
    frame_count = 0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=tot_frames, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize_dim)

        if normalize:
            frame = frame / 255.0 
            ## Here we won't reap the benefits of normalization until we do further calculations using the normalized image

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame * 255 if normalize else frame)
        frame_paths.append(frame_filename)

        frame_count += 1
        progress_bar.update(1)

    cap.release()
    progress_bar.close()
    return frame_paths

def augment_frame(frame):
    aug_choice = random.choice(['rotate', 'flip', 'brightness'])

    if aug_choice == 'rotate':
        angle = random.uniform(-15, 15)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        augmented_frame = cv2.warpAffine(frame, M, (w, h))
        
    elif aug_choice == 'flip':
        flip_code = random.choice([-1, 0, 1])
        augmented_frame = cv2.flip(frame, flip_code)
        
    elif aug_choice == 'brightness':
        value = random.randint(-50, 50)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        augmented_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
    return augmented_frame

def process_frame(frame_path, augmented_output_dir):
    frame = cv2.imread(frame_path)
    augmented_frame = augment_frame(frame)
    frame_filename = os.path.basename(frame_path)
    augmented_frame_path = os.path.join(augmented_output_dir, frame_filename)
    cv2.imwrite(augmented_frame_path, augmented_frame)

def main():
    video_path = 'cod.mp4'
    output_dir = 'extracted_frames'
    resize_dim = (224, 224)

    frame_paths = extract_frames(video_path, output_dir, resize_dim)
    print(f"Extracted and processed {len(frame_paths)} frames.")

    augmented_output_dir =  'augmented'
    os.makedirs(augmented_output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda frame_path: process_frame(frame_path, augmented_output_dir), frame_paths), total=len(frame_paths), desc="Augmenting frames"))

if __name__ == "__main__":
    main()
