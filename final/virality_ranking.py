# virality_analysis.py

import os
import cv2
import numpy as np
import logging
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from joblib import load
import re
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the VADER lexicon is downloaded
import nltk
nltk.download('vader_lexicon')

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_PIXEL_VALUE = 255
NO_OF_CHANNELS = 3
TIMESTEPS = 10

# Function to extract frames from a video clip
def extract_frames(video_path):
    frames_list = []
    videoObj = cv2.VideoCapture(video_path)
    logging.info(f"Extracting frames from video: {video_path}")
    while len(frames_list) < TIMESTEPS:
        success, image = videoObj.read()
        if not success:
            break
        resized_frame = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / MAX_PIXEL_VALUE
        frames_list.append(normalized_frame)
    videoObj.release()
    
    # Pad with last frame if needed to meet TIMESTEPS
    while len(frames_list) < TIMESTEPS:
        frames_list.append(frames_list[-1])
    
    return frames_list[:TIMESTEPS]

# Load pre-trained action detection model
def load_action_model(model_path):
    logging.info(f"Loading action detection model from: {model_path}")
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        logging.error(f"Error loading action detection model: {e}")
        raise e
    return model

# Predict actions in a video
def predict_actions(frames, model):
    features = np.array(frames).reshape(1, TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    logging.info(f"Predicting actions for frames with shape: {features.shape}")
    predictions = model.predict(features)
    return predictions.flatten()

# Perform sentiment analysis on text
def sentiment_analysis(text):
    vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer.polarity_scores(text)

# Preprocess text and combine with action predictions
def preprocess_transcript_and_actions(transcript, frames, action_model):
    # Sentiment analysis on the transcript
    sentiment_scores = sentiment_analysis(transcript)
    
    # Predict actions in the video frames
    action_predictions = predict_actions(frames, action_model)
    
    # Combine sentiment and action predictions into a single feature vector
    sentiment_df = pd.DataFrame([sentiment_scores])
    action_df = pd.DataFrame([action_predictions], columns=[f'action_{i}' for i in range(len(action_predictions))])
    
    combined_features = pd.concat([sentiment_df, action_df], axis=1)
    return combined_features

# Predict virality of a new clip
def predict_virality(transcript, video_path, virality_model, action_model):
    # Extract frames from the video
    frames = extract_frames(video_path)
    
    # Prepare features for virality prediction
    features = preprocess_transcript_and_actions(transcript, frames, action_model)
    
    # Predict virality score
    virality_score = virality_model.predict(features)
    return virality_score[0]

# Rank video clips based on virality scores
def rank_clips(clip_paths, transcripts, virality_model, action_model):
    ranked_clips = []
    for clip_path, transcript in zip(clip_paths, transcripts):
        score = predict_virality(transcript, clip_path, virality_model, action_model)
        ranked_clips.append((score, clip_path))
    
    # Sort clips by virality score in descending order
    ranked_clips.sort(reverse=True, key=lambda x: x[0])
    return ranked_clips
