import cv2
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames_multithreaded(video_path, num_threads=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def process_frame(frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame_index, gray
        return frame_index, None
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_frame, i) for i in range(total_frames)]
        frames = {i: f.result()[1] for i, f in enumerate(futures) if f.result()[1] is not None}
    
    cap.release()
    return frames

def calculate_histogram_difference(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def detect_shot_boundaries(video_path, method='sift', diff_threshold=50000, match_threshold=0.7, num_threads=4):
    """
    Detects shot boundaries in a video using either SIFT or frame differencing.

    Args:
        video_path (str): Path to the video file.
        method (str): Method for shot detection ('sift' or 'diff').
        diff_threshold (int): Threshold for frame differencing method.
        match_threshold (float): Threshold for SIFT method.
        num_threads (int): Number of threads to use for frame extraction.
    
    Returns:
        list: List of frame indices where shot boundaries were detected.
    """
    frames = extract_frames_multithreaded(video_path, num_threads=num_threads)
    if method == 'sift':
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    prev_frame = None
    prev_des = None
    shot_boundaries = []
    
    logging.info(f"Total frames in video: {len(frames)}")
    
    with tqdm(total=len(frames), desc="Detecting shot boundaries") as pbar:
        for frame_index in sorted(frames.keys()):
            gray = frames[frame_index]
            
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

def refine_shot_boundaries(video_path, initial_boundaries, threshold=0.5):
    """
    Refines shot boundaries based on SIFT feature matching.

    Args:
        video_path (str): Path to the video file.
        initial_boundaries (list): Initial list of shot boundaries.
        threshold (float): Threshold for refining shot boundaries.
    
    Returns:
        list: List of refined shot boundaries.
    """
    frames = extract_frames_multithreaded(video_path)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    refined_boundaries = []
    prev_des = None
    
    for frame_index in sorted(frames.keys()):
        gray = frames[frame_index]
        kp, des = sift.detectAndCompute(gray, None)
        
        if prev_des is not None:
            matches = bf.match(prev_des, des)
            distances = [m.distance for m in matches]
            
            if len(distances) > 0:
                avg_distance = sum(distances) / len(distances)
                if avg_distance > threshold:
                    if frame_index in initial_boundaries:
                        refined_boundaries.append(frame_index)
        
        prev_des = des
    
    return refined_boundaries

def main():
    video_path = "input_video.mp4"
    method = 'sift'  # Can be 'diff' for frame differencing or 'sift' for SIFT-based detection
    
    logging.info("Starting initial shot boundary detection")
    initial_boundaries = detect_shot_boundaries(video_path, method=method)
    
    logging.info("Starting shot boundary refinement")
    refined_boundaries = refine_shot_boundaries(video_path, initial_boundaries)
    
    print("Detected shot boundaries at frames:", refined_boundaries)

if __name__ == "__main__":
    main()
