import numpy as np
import cv2
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        std = np.std(gray)
        features.append([mean, std])
    
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(features)

def scrape_tiktok_hashtags(hashtags, num_videos=100):
    # Placeholder for actual TikTok scraping logic
    data = []
    for hashtag in hashtags:
        for i in range(num_videos):
            data.append({
                'description': f'{hashtag} video {i}',
                'likes': np.random.randint(100, 10000)
            })
    return data

def extract_features_from_clip(clip_path):
    features = extract_video_features(clip_path)
    return np.mean(features, axis=0)  # Use mean of features as a simple example

def rank_clips(clip_paths, model, scaler):
    features = [extract_features_from_clip(clip_path) for clip_path in clip_paths]
    features = np.array(features)
    features_scaled = scaler.transform(features)
    scores = model.predict(features_scaled)
    ranked_clips = sorted(zip(scores, clip_paths), reverse=True, key=lambda x: x[0])
    return ranked_clips

def train_virality_model(video_features_file, labels_file, model_file, scaler_file):
    X = np.load(video_features_file)
    y = np.load(labels_file)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVR()
    model.fit(X_scaled, y)
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    logging.info("Model trained and saved")

def load_model(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def main():
    # Training phase
    video_features_file = '/Users/kesinishivaram/FragsAI/video_features.npy'
    labels_file = '/Users/kesinishivaram/FragsAI/virality_labels.npy'
    model_file = 'virality_model.pkl'
    scaler_file = 'scaler.pkl'
    
    logging.info("Training virality model")
    train_virality_model(video_features_file, labels_file, model_file, scaler_file)
    
    # Ranking phase
    clip_paths = ["clips/clip_001.mp4", "clips/clip_002.mp4", "clips/clip_003.mp4"]  # Example clip paths
    model, scaler = load_model(model_file, scaler_file)
    
    logging.info("Ranking clips based on virality")
    ranked_clips = rank_clips(clip_paths, model, scaler)
    
    print("Ranked clips:")
    for score, clip_path in ranked_clips:
        print(f"Clip: {clip_path}, Score: {score}")

if __name__ == "__main__":
    main()