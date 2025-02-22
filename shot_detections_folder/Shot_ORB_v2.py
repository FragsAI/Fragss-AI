import cv2
import os
import gc
import numpy as np
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import moviepy
from PIL import Image
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.config import change_settings
from pathlib import Path

import preprocessing_final_v2 
from preprocessing_final_v2 import (adjust_sample_interval, determine_chunk_size, extract_frames_v2, frame_generator_v2, 
                                convert_to_grayscale_v2, apply_augmentations_v2, process_frames_v2) 

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

def detect_shot_boundaries_v2(frames, frame_indices):
    
    thresholds = dynamic_threshold_adjustment(frames)
    non_zero_threshold, distance_threshold, histogram_threshold = thresholds.values()

    shot_boundaries = []
    
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    
    for i in range(len(frames) - 1):
        prev_frame, frame = frames[i], frames[i + 1]
        gray_prev, gray = prev_frame, frame  

        # if convert_gray= True:
        #     # Convert frames to grayscale for feature matching
        #     gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # else:
        #     gray_prev, gray = prev_frame, frame           

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
        """
        Compares non zero counts, average of matching distances, histogram differences (correlation) between two frames with repective thresholds
        and if any two conditions are satisfied then detects that frame index as shot boundary (appends to list shot_boundaries).        
        """
        condition1 = non_zero_count > non_zero_threshold   # if non_zero_count > non_zero_threshold then change in scene
        condition2 = average_distance > distance_threshold # if average_distance > distance_threshold then change in scene
        condition3 = correlation < histogram_threshold     # if correlation < histogram_threshold then change in scene
        
        if sum([condition1, condition2, condition3]) >= 2:
            # shot_boundaries.append(i + 1)  # Frame index where scene change occurs
            shot_boundaries.append(frame_indices[i + 1])
            
    return shot_boundaries

def process_shot_detection_v2(frames_batch_folder, indices_batch_folder, output_file):
    shot_boundary_batch_npy_file_name = 'shot_boundaries_batch.npy' # .npy file to save a array of gray scaled frame
    final_shot_boundary_batch_dir = os.path.join(output_file,  shot_boundary_batch_npy_file_name)

    total_frames_with_shot_boundaries = 0
    batch_shot_boundaries = []
    
    batch_files = sorted([f for f in os.listdir(frames_batch_folder) if f.endswith(".npy")])
    indices_files = sorted([f for f in os.listdir(indices_batch_folder) if f.endswith(".npy")])
    logging.info(f"Processing {len(batch_files)} batches of frames for shot detection.")
    
    with tqdm(total=len(batch_files), desc="Detecting frames with shot boundary") as progress:
        for i, (batch_file, indices_file) in enumerate(zip(batch_files, indices_files)):
            batch_file_path = os.path.join(frames_batch_folder, batch_file)
            frames = np.load(batch_file_path) 

            indices_file_path = os.path.join(indices_batch_folder, indices_file)
            frame_indices = np.load(indices_file_path)
            
            shot_boundaries = detect_shot_boundaries_v2(frames, frame_indices)
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

def fetch_indices_batch(batch_num, indices_batch_folder):
    """
    Fetches shot boundary indices
    args:
    batch_num: batch number of which batch to fetch indices from
    indices_batch_folder: Path where shot boundary indices batches saved
    """
    indices_batch_file=(os.path.join(indices_batch_folder, f"Indices_batch_000{batch_num}.npy") 
                        if Path(os.path.join(indices_batch_folder, f"Indices_batch_000{batch_num}.npy")).is_file() 
                        else os.path.join(indices_batch_folder,f"shot_boundaries_batch_{batch_num}.npy"))
    print(f"Loading batch: {indices_batch_file}.", end= ' ')
    frames_indices_batch = np.load(indices_batch_file)
    print(f"No.of indices: {frames_indices_batch.shape[0]}")
    return frames_indices_batch
     

def time_stams_by_index(video_path):
    """
    Generates dictonary of frames and timestamps; frame numbers as key and time stamps as value
    args:
    video_path: your video file
    """
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    frame_wise_timestamps={}
    logging.info('Extracting time stamps...')
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            prev_time_stamp=cap.get(cv2.CAP_PROP_POS_MSEC)
            # print("frame no. " + str(frame_num) + " timestamp is: ", str(prev_time_stamp))
            frame_wise_timestamps[frame_num]=prev_time_stamp
        else:
            break
        frame_num += 1
    
    cap.release()
    logging.info(f"Process completed. {len(frame_wise_timestamps)} time stamps extracted")
    return frame_wise_timestamps
    
def generate_clip(video_path, output_dir, from_index ,start_time, end_time):
    """
    Generates video clips based shot boundary index
    """
    video_name = [ i for i in video_path.split('\\') if i.endswith(('.mp4','.avi'))][0][:-4]
    file_name = os.path.join(output_dir, f'{video_name}_from_{from_index}.mp4') 
    video = VideoFileClip(video_path)
    clip=video.subclip(start_time, end_time)
    logging.info(f"Generating clip from shot boundary {from_index}...")
    clip.write_videofile(file_name,codec="libx264")
    logging.info(f"Clip saved as '{file_name}'")

def main():
    video_path = r"C:\Users\video.mp4" #your video
    output_dir = r"C:\Users\FramesGenerated" # directory to save generated batches of frames
    indices_output_dir=r"C:\Users\IndicesBatches" # directory to save batches of frame indices
    gray_scaled_frames_dir = r"C:\Users\GrayFrames" # directory to save gray scaled batches of frames
    
    shot_boundary_batch_dir=r"C:\Users\ShotBoundaryBatches" # directory to save batches of frames indices with shot boundary
    
    sample_interval = adjust_sample_interval(video_path) 
    chunk_size = determine_chunk_size() #No.of frames per batch
    
    # Extracting frames in batches and save in output_dir
    extract_frames_v2(video_path=video_path, frames_output_folder=output_dir,indices_output_folder=indices_output_dir, sample_interval=sample_interval, downscale_factor=2, batch_size=chunk_size)

    #Processing frames to convert to gray scale or augment and resizing and save in batches in gray_scaled_frames_dir
    process_frames_v2(batch_folder= output_dir,  output_file = gray_scaled_frames_dir, resize_dim=(224, 224), augment=False)

    # Detect frames with shot boundaries, get their indices and save batch-wise
    shot_boundaries = process_shot_detection_v2(frames_batch_folder = gray_scaled_frames_dir, indices_batch_folder = indices_output_dir, output_file = shot_boundary_batch_dir)

    #cacluate frame wise timestamps 
    timestamps=time_stams_by_index(video_path)    
    
    #Fetch list of shot boundary indices of batch
    batch_num=1 # adjust to required batch number
    shot_boundary_indices= fetch_indices_batch(batch_num, indices_batch_folder=shot_boundary_batch_dir)    
    
    #calculate start and end time for clip generation
    start_index = 0 #adjust as per requirement
    end_index = 10 #adjust as per requirement
    starttime=timestamps[shot_boundary_indices[start_index]]/1000
    endtime=timestamps[shot_boundary_indices[end_index]]/1000 
    
    output_dir_clips = r"C:\Users\ShotBoundaryClips" #Dirctory to save clips #end_time 120+start time  for 2 min clip
    generate_clip(video_path, output_dir=output_dir_clips, from_index=shot_boundaries[start_index],start_time=starttime, end_time=starttime+120)

if __name__ =="__main__":
    main()