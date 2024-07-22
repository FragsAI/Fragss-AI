import cv2
import numpy as np
import os
import logging
from moviepy.editor import VideoFileClip
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime as dt
from sklearn.metrics import classification_report
from scipy.stats import variation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
TIMESTEPS = 10  # Number of frames to consider in each sequence
CLASS_CATEGORIES_LIST = ["Nunchucks", "Punch"]
MAX_PIXEL_VALUE = 255
BATCH_SIZE = 100
NO_OF_CHANNELS = 3
NO_OF_CLASSES = len(CLASS_CATEGORIES_LIST)
MODEL_PATH = '/Users/kesinishivaram/FragsAI/Model___Date_Time_2024_07_16__13_48_36___Loss_2.2341885566711426___Accuracy_0.774193525314331.h5'  # Adjust path to your pre-trained model

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
    return np.array(frames_list[:TIMESTEPS])

# Function to extract audio from video
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    audio, sr = librosa.load(audio_path, sr=None)
    os.remove(audio_path)
    return audio, sr

# Function to find loudest moments in audio
def find_loudest_moments(audio, sr, num_clips=15, clip_length=15):
    frame_length = sr * clip_length
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length)[0]
    loudest_indices = np.argsort(rms)[-num_clips:]
    loudest_times = loudest_indices * clip_length
    return loudest_times

# Function to segment video
def segment_video(video_path, events, segment_duration=15):
    video = VideoFileClip(video_path)
    clips = []
    for event in events:
        start_time = max(0, event - segment_duration / 2)
        end_time = min(video.duration, event + segment_duration / 2)
        if end_time - start_time < segment_duration:
            start_time = max(0, end_time - segment_duration)
        clip = video.subclip(start_time, end_time)
        clips.append(clip)
    return clips

# Function to save clips
def save_clips(clips, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, clip in enumerate(clips):
        output_path = os.path.join(output_dir, f"clip_{i + 1}.mp4")
        clip.write_videofile(output_path)

# Load pre-trained action detection model
logging.info(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)

# Function to predict actions in frames
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

# Main function to process video
def main(video_path, output_dir="clips", num_clips=10, clip_length=15):
    audio, sr = extract_audio(video_path)
    loudest_times = find_loudest_moments(audio, sr, num_clips=num_clips, clip_length=clip_length)
    clips = segment_video(video_path, loudest_times, segment_duration=clip_length)
    save_clips(clips, output_dir)
    clip_scores = process_videos_in_folder(output_dir)
    
    for clip_path, score in clip_scores:
        logging.info(f"Clip: {clip_path}, Virality Score: {score}")

if __name__ == "__main__":
    video_path = '/Users/kesinishivaram/FragsAI/Fragss-AI/cod.mp4'  # Adjust path to your video
    main(video_path)
