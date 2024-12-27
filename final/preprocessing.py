# preprocessing.py

import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
from skimage.util import random_noise
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Tuple, Dict
import queue
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, video_path: str, output_folder: str, 
                 batch_size: int = 32, max_resolution: Tuple[int, int] = (1280, 720),
                 num_threads: int = 4):
        """
        Initialize the video processor with optimized settings.
        
        Args:
            video_path: Path to the video file
            output_folder: Directory to save processed frames
            batch_size: Number of frames to process in each batch
            max_resolution: Maximum resolution to resize video frames to
            num_threads: Number of threads for parallel processing
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.max_resolution = max_resolution
        self.num_threads = num_threads
        self.frame_queue = queue.Queue(maxsize=100)
        self.batch_queue = queue.Queue(maxsize=10)
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.original_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        # Calculate target resolution while maintaining aspect ratio
        self.target_resolution = self._calculate_target_resolution()
        
    def _calculate_target_resolution(self) -> Tuple[int, int]:
        """Calculate the target resolution maintaining aspect ratio."""
        orig_w, orig_h = self.original_resolution
        max_w, max_h = self.max_resolution
        
        # Calculate scaling factor
        scale = min(max_w/orig_w, max_h/orig_h)
        
        # Calculate new dimensions
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        return (new_w, new_h)
    
    def extract_frames(self, frame_rate: int = 1) -> List[str]:
        """
        Extract frames from video with optimized batch processing.
        
        Args:
            frame_rate: Extract every nth frame
            
        Returns:
            List of paths to saved frames
        """
        frame_paths = []
        
        # Start frame reading thread
        read_thread = threading.Thread(target=self._read_frames, args=(frame_rate,))
        read_thread.daemon = True
        read_thread.start()
        
        # Start batch processing threads
        process_threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._process_frame_batch)
            t.daemon = True
            t.start()
            process_threads.append(t)
        
        # Start batch saving thread
        save_thread = threading.Thread(target=self._save_frame_batch)
        save_thread.daemon = True
        save_thread.start()
        
        # Wait for all frames to be processed
        read_thread.join()
        for t in process_threads:
            t.join()
        save_thread.join()
        
        self.cap.release()
        return frame_paths
    
    def _read_frames(self, frame_rate: int):
        """Read frames from video and add to queue."""
        frame_count = 0
        with tqdm(total=self.total_frames//frame_rate, desc="Reading frames") as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_rate == 0:
                    self.frame_queue.put(frame)
                    pbar.update(1)
                    
                frame_count += 1
                
        # Signal completion
        self.frame_queue.put(None)
    
    def _process_frame_batch(self):
        """Process frames in batches with resizing and augmentation."""
        batch = []
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                if batch:
                    self.batch_queue.put(batch)
                self.batch_queue.put(None)
                break
                
            # Resize frame
            frame = cv2.resize(frame, self.target_resolution)
            
            # Apply augmentations
            frame = self._augment_frame(frame)
            
            batch.append(frame)
            if len(batch) >= self.batch_size:
                self.batch_queue.put(batch)
                batch = []
    
    def _save_frame_batch(self):
        """Save processed frame batches to disk."""
        frame_count = 0
        active_threads = self.num_threads
        
        while active_threads > 0:
            batch = self.batch_queue.get()
            if batch is None:
                active_threads -= 1
                continue
                
            for frame in batch:
                frame_path = os.path.join(self.output_folder, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_count += 1
    
    def _augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply advanced augmentations to the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Augmented frame
        """
        # Apply random augmentations with probabilities
        if random.random() < 0.5:
            frame = cv2.flip(frame, 1)  # Horizontal flip
            
        if random.random() < 0.3:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Gaussian blur
            
        if random.random() < 0.3:
            # Random brightness adjustment
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-30, 30)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
        if random.random() < 0.2:
            # Random contrast adjustment
            frame = frame.astype(float)
            mean = np.mean(frame)
            frame = (frame - mean) * random.uniform(0.8, 1.2) + mean
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
        if random.random() < 0.2:
            # Random noise
            noise_type = random.choice(['gaussian', 'speckle', 's&p'])
            frame = random_noise(frame, mode=noise_type)
            frame = np.array(255 * frame, dtype='uint8')
            
        if random.random() < 0.3:
            # Random cropping
            h, w = frame.shape[:2]
            crop_percent = random.uniform(0.8, 1.0)
            crop_w = int(w * crop_percent)
            crop_h = int(h * crop_percent)
            x = random.randint(0, w - crop_w)
            y = random.randint(0, h - crop_h)
            frame = frame[y:y+crop_h, x:x+crop_w]
            frame = cv2.resize(frame, (w, h))
            
        return frame

def preprocess_video(video_path: str, output_folder: str, frame_rate: int = 1,
                    batch_size: int = 32, max_resolution: Tuple[int, int] = (1280, 720),
                    num_threads: int = 4) -> List[str]:
    """
    Main function to preprocess a video with optimized settings.
    
    Args:
        video_path: Path to the video file
        output_folder: Directory to save processed frames
        frame_rate: Extract every nth frame
        batch_size: Number of frames to process in each batch
        max_resolution: Maximum resolution to resize video frames to
        num_threads: Number of threads for parallel processing
        
    Returns:
        List of paths to saved frames
    """
    processor = VideoProcessor(
        video_path=video_path,
        output_folder=output_folder,
        batch_size=batch_size,
        max_resolution=max_resolution,
        num_threads=num_threads
    )
    
    return processor.extract_frames(frame_rate)

if __name__ == "__main__":
    # Example usage
    video_path = "path/to/your/video.mp4"
    output_dir = "processed_frames"
    
    frame_paths = preprocess_video(
        video_path=video_path,
        output_folder=output_dir,
        frame_rate=5,  # Extract every 5th frame
        batch_size=32,
        max_resolution=(1280, 720),  # 720p maximum resolution
        num_threads=4
    )
    
    print(f"Processed {len(frame_paths)} frames")
