import cv2
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import preprocessing_final # import preprocessing_final(1)
from preprocessing_final import (adjust_sample_interval, determine_chunk_size, frame_generator, 
                                convert_to_grayscale, apply_augmentations, process_frames) 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_histogram_difference(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def dynamic_threshold_adjustment(frames, convert_to_gray=False):
    """Dynamically calculates thresholds for shot boundary detection based on frame differences."""
    if len(frames) < 2:
        raise ValueError("At least two frames are required to compute thresholds.")
        
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)    
    
    non_zero_counts = []
    avg_distances = []
    hist_correlations = []    
    
    for i in range(len(frames) - 1):
        prev_frame, frame = frames[i], frames[i + 1] 
        
        # Compute Non-Zero Pixel Differences
        diff = cv2.absdiff(prev_frame, frame)
        non_zero_counts.append(np.count_nonzero(diff)) 
        
        # Calculate ORB Feature Matching Distances
        if convert_to_gray == True:
            gray1, gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = prev_frame, frame            
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)        
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            matches = bf.match(des1, des2)
            distances = [m.distance for m in matches]
            avg_distance = np.mean(distances) if distances else 0
            avg_distances.append(avg_distance)
        else:
            avg_distances.append(0)  # No valid matches found 
            
        # Calculate Histogram Differences
        hist_correlation = calculate_histogram_difference(gray1, gray2)
        hist_correlations.append(hist_correlation)    
        
    # Calculate Thresholds
    non_zero_mean, non_zero_std = np.mean(non_zero_counts), np.std(non_zero_counts)
    dist_mean, dist_std = np.mean(avg_distances), np.std(avg_distances)
    hist_mean, hist_std = np.mean(hist_correlations), np.std(hist_correlations)
    
    # dynamically adjusted thresholds
    non_zero_threshold = int(np.round(non_zero_mean + (non_zero_std / 2)))
    distance_threshold = np.round((dist_mean + dist_std),3)
    histogram_threshold = np.round(hist_mean - (2 * hist_std),3)
    
    thresholds = { "non_zero_threshold": non_zero_threshold,
                  "distance_threshold": distance_threshold,
                  "histogram_threshold": histogram_threshold }
    
    return thresholds

def detect_shot_boundaries(frames):
    
    thresholds = dynamic_threshold_adjustment(frames)
    non_zero_threshold, distance_threshold, histogram_threshold = thresholds.values()

    shot_boundaries = []
    
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    for i in range(len(frames) - 1):
        prev_frame, frame = frames[i], frames[i + 1]
        gray_prev, gray = prev_frame, frame  

        # Calculate count of non-zero pixel differences between previous frame and current frame
        non_zero_count = np.count_nonzero(cv2.absdiff(gray_prev, gray)) #gray_prev- previous frame, gray- current frame
        
        # ORB feature matching
        keypoints1, descriptors1 = orb.detectAndCompute(gray_prev, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray, None)
       
        if descriptors1 is not None and descriptors2 is not None:
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            average_distance = np.mean([m.distance for m in matches]) if matches else 0
        else:
            average_distance = 0
        
        # Calculate Histogram difference (correlation)
        correlation = calculate_histogram_difference(gray_prev, gray)
        
        # Check scene change conditions
        '''
        Compares non zero counts, average of matching distances, histogram differences (correlation) between two frames with repective thresholds
        and if any two conditions are satisfied then detects that frame index as shot boundary (appends to list shot_boundaries).        
        '''
        condition1 = non_zero_count > non_zero_threshold   # if non_zero_count > non_zero_threshold then change in scene
        condition2 = average_distance > distance_threshold # if average_distance > distance_threshold then change in scene
        condition3 = correlation < histogram_threshold     # if correlation < histogram_threshold then change in scene
        
        if sum([condition1, condition2, condition3]) >= 2:
            shot_boundaries.append(i + 1)  # Frame index where scene change occurs
            
    return shot_boundaries

def process_shot_detection(batch_folder, output_file):
    shot_boundary_batch_npy_file_name = 'shot_boundaries_batch.npy' # .npy file to save a array of gray scaled frame
    final_shot_boundary_batch_dir = os.path.join(output_file,  shot_boundary_batch_npy_file_name)

    total_frames_with_shot_boundaries = 0
    batch_shot_boundaries = []
    
    batch_files = sorted([f for f in os.listdir(batch_folder) if f.endswith(".npy")])
    logging.info(f"Processing {len(batch_files)} batches of frames for shot detection.")
    
    with tqdm(total=len(batch_files), desc="Detecting frames with shot boundary") as progress:
        for i, batch_file in enumerate(batch_files):
            batch_file_path = os.path.join(batch_folder, batch_file)
            frames = np.load(batch_file_path) 

            #Detect shot boundary indices
            shot_boundaries = detect_shot_boundaries(frames)
            batch_shot_boundaries.extend(shot_boundaries)
            batch_shot_boundaries_array = np.array(batch_shot_boundaries, dtype = np.int32)
            
            total_frames_with_shot_boundaries += len(batch_shot_boundaries)
            progress.update(1)
    
            temp_output = final_shot_boundary_batch_dir.replace(".npy", f"_{i}.npy")
            np.save(temp_output, np.array(batch_shot_boundaries_array, dtype=np.int32))
            batch_shot_boundaries.clear()
            gc.collect()

    logging.info(f"{total_frames_with_shot_boundaries} Indices of frames with shot boundaries saved to {output_file}")
    if batch_shot_boundaries:
        np.save(output_file, np.array(batch_shot_boundaries_array, dtype=np.int32))
        logging.info(f"Indices of frames with shot boundaries saved to {output_file}")

def main():
    # Generate and Extract frames from video path
    video_path = r"video.mp4" # Your video path
    output_dir = r"\Frames" # directory to save generated batches of frames
    gray_scaled_frames_dir = r"\GrayFrames" # directory to save gray scaled batches of frames
    augmented_frames_dir = r"\AugmentedFrames" # directory to save augmented batches of frames if augmentation required inplace of gray scale
    shot_boundary_batch_dir = r"\shot_boundary_batch" # directory to save gray scaled batches of frames

    sample_interval = adjust_sample_interval(video_path) #No.of frames to skip in between
    chunk_size = determine_chunk_size() #No.of frames per batch
    
    # Extracting frames in batches and save in output_dir
    extract_frames(video_path, output_dir, sample_interval, downscale_factor=2, batch_size=chunk_size)
    
    #Processing frames to convert to gray scale or augment and resizing and save in batches in gray_scaled_frames_dir
    process_frames(batch_folder= output_dir,  output_file = gray_scaled_frames_dir, resize_dim=(224, 224), augment=False)

    # Detect frames with shot boundaries, get their indices and save batch-wise
    shot_boundaries = process_shot_detection(batch_folder = gray_scaled_frames_dir, output_file = shot_boundary_batch_dir)

if __name__ =="__main__":
    main()
