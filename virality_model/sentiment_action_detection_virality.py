import os
import cv2
import numpy as np
import logging
import re
import pandas as pd
import joblib
import ssl
from tensorflow.keras.models import load_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_PIXEL_VALUE = 255
NO_OF_CHANNELS = 3
TIMESTEPS = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SSL settings for NLTK
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

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
action_model_path = '/Users/kesinishivaram/FragsAI/action_detection_model/model.h5'
logging.info(f"Loading action detection model from: {action_model_path}")
try:
    action_model = load_model(action_model_path, compile=False)
except Exception as e:
    logging.error(f"Error loading action detection model: {e}")
    exit(1)

# Function to predict actions in a video
def predict_actions(frames):
    features = np.array(frames).reshape(1, TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    logging.info(f"Predicting actions for frames with shape: {features.shape}")
    predictions = action_model.predict(features)
    return predictions

# Function to perform sentiment analysis
def sentiment_analysis(text: str) -> dict:
    vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer.polarity_scores(text)

# Function to load and concatenate CSVs
def load_and_concatenate_csvs(folder_path: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

# Function to preprocess data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['subtitle_sentiment'] = df['subtitles'].apply(lambda x: sentiment_analysis(x) if pd.notnull(x) else {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
    df['top_comment_sentiment'] = df['top_comment'].apply(lambda x: sentiment_analysis(x) if pd.notnull(x) else {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})

    df = pd.concat([df.drop(['subtitle_sentiment', 'top_comment_sentiment'], axis=1),
                    df['subtitle_sentiment'].apply(pd.Series),
                    df['top_comment_sentiment'].apply(pd.Series)], axis=1)

    df['duration_seconds'] = df['duration'].apply(lambda x: int(re.search(r'\d+S', x).group(0).replace('S', '')) if 'S' in x else 0)
    
    df['virality_score'] = (df['views'] - df['views'].min()) / (df['views'].max() - df['views'].min()) * 100
    
    return df

# Function to prepare data for training
def prepare_data(df: pd.DataFrame, target_col: str):
    features = df.drop(columns=[target_col, 'id', 'subtitles', 'top_comment', 'duration'])
    target = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    return model

# Function to preprocess transcript and combine with action predictions
def preprocess_transcript_and_actions(srt_file_path: str, video_path: str, model):
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()
    
    transcript = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', srt_content)
    transcript = re.sub(r'\n\d+\n', '\n', transcript)
    transcript = transcript.replace('\n', ' ').strip()
    
    sentiment_scores = sentiment_analysis(transcript)
    
    # Extract and predict actions from the video
    frames = extract_frames(video_path)
    action_predictions = predict_actions(frames).flatten()
    
    # Combine sentiment and action predictions into a single feature vector
    sentiment_df = pd.DataFrame([sentiment_scores])
    action_df = pd.DataFrame([action_predictions], columns=[f'action_{i}' for i in range(len(action_predictions))])
    
    combined_features = pd.concat([sentiment_df, action_df], axis=1)
    
    # Get feature names from the model to ensure the correct order
    feature_names = model.feature_names_in_
    combined_features = combined_features[feature_names]
    
    return combined_features

# Function to predict virality of a new clip
def predict_virality(srt_file_path: str, video_path: str, model_filename: str = 'virality_model.pkl'):
    model = joblib.load(model_filename)
    features = preprocess_transcript_and_actions(srt_file_path, video_path, model)
    
    virality_score = model.predict(features)
    
    return virality_score[0]

if __name__ == "__main__":
    # TRAINING AND SAVING THE MODEL
    folder_path = "/Users/kesinishivaram/FragsAI/youtube_data"
    
    df = load_and_concatenate_csvs(folder_path)
    
    df = preprocess_data(df)
    
    target_col = 'virality_score'
    
    X_train, X_test, y_train, y_test = prepare_data(df, target_col)
    
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    model_filename = 'virality_model.pkl'
    joblib.dump(model, model_filename)
    
    # PREDICTING VIRALITY OF NEW CLIPS BASED ON VIDEO AND TRANSCRIPT FILE
    srt_file_path = '/Users/kesinishivaram/FragsAI/clips/sub-clips/clip_10_aspect_ratio.mp4.en.srt'  # Replace with your .srt file path
    video_path = '/Users/kesinishivaram/FragsAI/clips/sub-clips/clip_10_aspect_ratio.mp4'  # Replace with your video file path
    
    virality_score = predict_virality(srt_file_path, video_path) * 10000
    print(f'Predicted Virality Score: {virality_score}')
