import cv2
import numpy as np
import os

def extract_sift_features(frame):
    sift = cv2.SIFT_create()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    return keypoints, descriptors

def extract_surf_features(frame):
    surf = cv2.SURF_create()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = surf.detectAndCompute(gray_frame, None)
    return keypoints, descriptors

def preprocess_frame(frame, resize_dim):
    # Resize the frame
    resized_frame = cv2.resize(frame, resize_dim)
    
    # Normalize the frame
    normalized_frame = cv2.normalize(resized_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return normalized_frame

def main(video_path, output_dir, resize_dim=(640, 480), step_size=2):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = resize_dim[0]
    frame_height = resize_dim[1]
    
    prev_descriptors = None
    scene_changes = []
    frame_number = 0
    min_frames_between_changes = int(3 * fps)  # Minimum frames corresponding to 3 seconds
    
    # List to store the preprocessed frames
    preprocessed_frames = []
    original_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % step_size != 0:
            frame_number += 1
            continue

        # Preprocess the frame (includes resizing and normalization)
        preprocessed_frame = preprocess_frame(frame, resize_dim)
        preprocessed_frames.append(preprocessed_frame)  # Store the preprocessed (resized and normalized) frame
        original_frames.append(frame)  # Store the original frame for final video writing
        
        # Extract SIFT features for the current frame
        _, curr_descriptors = extract_sift_features(preprocessed_frame)
        
        if prev_descriptors is not None:
            # Match SIFT features between current and previous frames
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(prev_descriptors, curr_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate the difference based on the number of good matches
            good_matches = [m for m in matches if m.distance < 0.75 * np.mean([m.distance for m in matches])]
            feature_diff = len(good_matches) / max(len(prev_descriptors), len(curr_descriptors))
            
            # If the feature difference is below a threshold, consider it a scene change
            if feature_diff < 0.1:  # Adjust this threshold based on your requirements
                if not scene_changes or (frame_number - scene_changes[-1] >= min_frames_between_changes):
                    scene_changes.append(frame_number)
                
        prev_descriptors = curr_descriptors
        frame_number += 1
        
    cap.release()
    
    # Adding the end frame number
    scene_changes.append(frame_number)
    
    print("Scene changes detected at frames:", scene_changes)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    current_scene = 0
    start_frame = 0

    for i, change_frame in enumerate(scene_changes):
        out = cv2.VideoWriter(
            os.path.join(output_dir, f'scene_{current_scene}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        
        for frame_idx in range(start_frame, change_frame):
            if frame_idx < len(original_frames):
                out.write(original_frames[frame_idx])
        
        out.release()
        current_scene += 1
        start_frame = change_frame
    
    print("Video segmentation completed. Segments saved in:", output_dir)

if __name__ == "__main__":
    video_path = "C:\Users\paras\OneDrive\Documents\Frags AI Code\cod.mp4"  # Replace with your video file path
    output_dir = "./rand2"  # Replace with your desired output directory
    main(video_path, output_dir)
