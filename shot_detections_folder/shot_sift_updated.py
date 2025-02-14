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
video_path = r"C:\Users\chimm\Frags AI\FragsAIVideosAndAudios\Fortnite Battle Royale.mp4" # change to your video path

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

def process_frame(frame_index):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # logging.info(f" Processing frames of the video begins.")
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_index, gray = frame_index, gray
    else:
        frame_index, gray = frame_index, None
        
    cap.release()    
    return frame_index, gray
    
def extract_frames_multithreaded(video_path, num_threads=4, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip=adjust_sample_interval(video_path)
    frame_indices = list(range(0, total_frames, frame_skip))
    cap.release()
    batch_size = determine_chunk_size()
    no_of_batches = round(len(frame_indices)/batch_size)
    logging.info(f"No.of batches: {no_of_batches} | Batch size: {batch_size} | Total frames: {total_frames}| Frames to skip: {frame_skip}")
    start = 0
    end = batch_size
    frames_batches = []
    try:
        with tqdm(total = no_of_batches, desc = 'Processing frames', unit= ' batches') as progress:
            for batch_num in range(0, no_of_batches+1):
                batch = frame_indices[start:end]
            
                with ThreadPoolExecutor(max_workers=4) as executor:
                     batch = list(executor.map(process_frame, batch))
                frames_batches.append(batch) 
                
                start = end
                end += batch_size
                batch_size = end-start
        
                progress.update(1)
                
        # print(f"Process completed. Converted {len(frames_batches)} batches of frames to gray scale.")     
        
        ret_frames = []  
        for batch in frames_batches:
            ret_frames.extend(batch)
            
        gray_frames=[]
        gray_frames_indices=[]
        gray_frames_indices_with_None_frame = []
        for ret_frame in ret_frames:
            if ret_frame[1] is not None:
                frame=ret_frame[1]
                gray_frames.append(frame)
                gray_frames_indices.append(ret_frame[0])
            else:
                gray_frames_indices_with_None_frame.append(ret_frame[0])
         
        # print(f" Process completed. Converted {len(gray_frames)} frames ({len(frames_batches)} batches) to gray scale.")
        logging.info(f"Processing completed. Total {len(gray_frames)} frame converted to gray scale.")
          
    except KeyboardInterrupt as kie:         
        # print(f"Process Interrupted!! Able to process {len(gray_frames)} frames.")
        logging.info(f"Process Interrupted!! Able to process {len(gray_frames)} frames.")
                         
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     futures = [executor.submit(process_frame, i) for i in frame_indices]
    #     frames = {i: f.result()[1] for i, f in zip(frame_indices, futures) if f.result()[1] is not None

    # logging.info(f" Processing completed. Total {len(gray_frames)} frame converted to gray scale.\n")
    return gray_frames
    

def calculate_histogram_difference(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# frames = extract_frames_multithreaded(video_path, num_threads=num_threads)
def detect_shot_boundaries(video_path=None,frames=None, method='sift', extracted_frames=True,
                           diff_threshold=50, match_threshold=0.7, num_threads=4, refine_boundaries = False, initial_boundaries=None):
    '''
    Takes processed frames to detect/refine shot boundaries.
    
    args:
    frames: Uses these frames to detect shot boundaries; Need extracted_frames to be True
    video_path: If frames and extracted_frames is None then need video path to extract and process frames
                If frames are passed and extracted_frames is True then need not pass video
    threshold: Threshold to pass for frames to be a shot boundary   
    num_threads: Number of workers
    refined_boundaries: If False then just detect shot boundaries. If True then refine shot boundaries, need initial shot boundaries.  
    initial_boundaries: List of initial shot boundaries to refine.
    
    '''
    if extracted_frames == False and frames == None:
        frames = extract_frames_multithreaded(video_path, num_threads=num_threads)

    if method == 'sift':
        method_used = 'SIFT'
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == 'orb':
        method_used = 'ORB'
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_frame = None
    prev_des = None
    
    if refine_boundaries==True:
        description="Refining shot boundaries"
        refined_shot_boundaries = []
    else:
        description="Detecting shot boundaries"
        shot_boundaries = []
    

    # logging.info(f"Total frames in video: {len(frames)}")
    if refine_boundaries == False:
        logging.info(f"Shot boundary detection: Total extracted frames- {len(frames)} | method- {method_used}")
    else:
        logging.info(f"Refining shot boundaries: Total extracted frames- {len(frames)} | Total initial boundaries- {len(initial_boundaries)} | method- {method_used}")
  
    with tqdm(total=len(frames), desc=description) as pbar:
        # for frame_index in sorted(frames.keys()):
        for frame_index in range(0, len(frames)):
            gray = frames[frame_index]

            if method == 'diff':
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    non_zero_count = np.count_nonzero(diff)
                    if non_zero_count > diff_threshold and refine_boundaries == False and initial_boundaries is None:
                        shot_boundaries.append(frame_index)
                    elif non_zero_count > diff_threshold and refine_boundaries == True and frame_index in initial_boundaries:
                        refined_shot_boundaries.append(frame_index)
                            
                prev_frame = gray

            elif method == 'sift':
                kp, des = sift.detectAndCompute(gray, None)
                if prev_des is not None:
                    matches = bf.match(prev_des, des)
                    distances = [m.distance for m in matches]
                    if len(distances) > 0:
                        avg_distance = sum(distances) / len(distances)
                        if avg_distance > match_threshold and refine_boundaries == False and initial_boundaries is None:
                            shot_boundaries.append(frame_index)
                        elif avg_distance > match_threshold and refine_boundaries == True and frame_index in initial_boundaries:
                            refined_shot_boundaries.append(frame_index)

            elif method == 'orb':
                kp, des = orb.detectAndCompute(gray, None)
                if prev_des is not None:
                    matches = bf.match(prev_des, des)
                    distances = [m.distance for m in matches]
                    if len(distances) > 0:
                        avg_distance = sum(distances) / len(distances)
                        if avg_distance > match_threshold and refine_boundaries == False and initial_boundaries is None:
                            shot_boundaries.append(frame_index)
                        elif avg_distance > match_threshold and refine_boundaries == True and frame_index in initial_boundaries:
                            refined_shot_boundaries.append(frame_index)
            prev_des = des

            pbar.update(1)
    if refine_boundaries == True:
        logging.info(f"Process completed. Refined {len(refined_shot_boundaries)} shot boundaries.") 
        return refined_shot_boundaries
    else:
        logging.info(f"Process completed. Detected {len(shot_boundaries)} shot boundaries.")
        return shot_boundaries

def main():
    video_path = "input_video.mp4"
    method = 'sift'  # Can be 'diff' for frame differencing or 'sift' for SIFT-based detection or 'orb' for ORB-based detection
    num_threads = 4

    logging.info("Starting frames extraction")
    frames = extract_frames_multithreaded(video_path, num_threads=num_threads)    

    logging.info("Starting initial shot boundary detection")
    initial_boundaries = detect_shot_boundaries(frames=frames, method=method, num_threads=num_threads)

    logging.info("Starting shot boundary refinement")
    refined_boundaries = detect_shot_boundaries(initial_boundaries=initial_boundaries, frames= frames, method=method, num_threads=num_threads, refine_boundaries = True)


    print("Detected shot boundaries at frames:", refined_boundaries)

if __name__ == "__main__":
    main()
