import cv2
import os
import numpy as np

def preprocess_video(video_path, output_dir, resize_shape=(224, 224), augment_data=False):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return 0

    frame_count = 0

    # Extract and preprocess each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame (resize, convert to grayscale, normalize)
        processed_frame = preprocess_frame(frame, resize_shape)

        # Data augmentation: horizontal flipping
        # Set augement_data to True if you want to flip it
        if augment_data:
            processed_frame = cv2.flip(processed_frame, 1)
        
        # Save preprocessed frame
        frame_name = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, processed_frame)
        frame_count += 1

    cap.release()
    return frame_count

def preprocess_frame(frame, resize_shape=(224, 224)):

    # Resize frame
    resized_frame = cv2.resize(frame, resize_shape)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize frame
    processed_frame = cv2.normalize(gray_frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return processed_frame

if __name__ == "__main__":

    # Change video_path to the path for your video
    video_path = "/Users/kesinishivaram/FragsAI/Fragss-AI/cod.mp4"  # Path to input video
    output_dir = "preprocessed_frames"  # Directory to save preprocessed frames

    # Preprocess video
    frame_count = preprocess_video(video_path, output_dir)

    print(f"Preprocessing complete. {frame_count} frames processed.")
