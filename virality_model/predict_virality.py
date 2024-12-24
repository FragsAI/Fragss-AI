import os
import cv2
import numpy as np
import logging
from tensorflow.keras.models import load_model
from scipy.stats import variation

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_PIXEL_VALUE = 255
NO_OF_CHANNELS = 3
TIMESTEPS = 10 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract frames from a video
def extract_frames(video_path):
    frames_list = []
    videoObj = cv2.VideoCapture(video_path)
    logging.info(f"Extracting frames from video: {video_path}")
    while True:
        success, image = videoObj.read()
        if not success:
            break
        resized_frame = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / MAX_PIXEL_VALUE
        frames_list.append(normalized_frame)
    videoObj.release()
    while len(frames_list) < TIMESTEPS:
        frames_list.append(frames_list[-1])
    return frames_list[:TIMESTEPS]

# Load pre-trained action detection model
model_path = '/Users/kesinishivaram/FragsAI/Model___Date_Time_2024_07_13__17_00_43___Loss_0.12093261629343033___Accuracy_0.9838709831237793.h5'
logging.info(f"Loading model from: {model_path}")
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# Function to predict actions in a video
def predict_actions(frames):
    features = np.array(frames).reshape(1, TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    logging.info(f"Predicting actions for frames with shape: {features.shape}")
    predictions = model.predict(features)
    return predictions

# Function to assess video quality
def assess_video_quality(frames):
    brightness_scores = []
    blur_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor((frame * MAX_PIXEL_VALUE).astype('uint8'), cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_scores.append(brightness)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(laplacian_var)
    
    mean_brightness = np.mean(brightness_scores)
    mean_blur = np.mean(blur_scores)
    
    return mean_brightness, mean_blur

# Function to calculate virality score
def calculate_virality(predictions, quality_metrics):
    mean_confidence = np.mean(np.max(predictions, axis=1))
    variance_confidence = variation(np.max(predictions, axis=1))
    low_confidence_penalty = np.sum(np.max(predictions, axis=1) < 0.5)
    
    mean_brightness, mean_blur = quality_metrics
    
    score = (mean_confidence * 100) - (variance_confidence * 10) - (low_confidence_penalty * 5)
    score += mean_brightness * 0.01
    score -= mean_blur * 0.01
    
    return score

# Function to normalize scores
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) * 99 + 1 for score in scores]
    return normalized_scores

# Process each video in the given folder
def process_videos_in_folder(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_scores = {}
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        frames = extract_frames(video_path)
        predictions = predict_actions(frames)
        quality_metrics = assess_video_quality(frames)
        virality_score = calculate_virality(predictions, quality_metrics)
        video_scores[video_file] = virality_score
    
    scores = list(video_scores.values())
    normalized_scores = normalize_scores(scores)
    
    for i, video_file in enumerate(video_scores):
        video_scores[video_file] = normalized_scores[i]
    
    sorted_video_scores = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_video_scores

# Main function
def main():
    folder_path = '/Users/kesinishivaram/FragsAI/clips'
    sorted_video_scores = process_videos_in_folder(folder_path)
    
    for video_file, score in sorted_video_scores:
        logging.info(f"Video: {video_file}, Virality Score: {score}")

if __name__ == "__main__":
    main()
