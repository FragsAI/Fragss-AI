import os
import cv2
import numpy as np
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
from tqdm import tqdm
import concurrent.futures
from tensorflow.keras.models import load_model
from scipy.special import softmax
from action_detection import extract_features

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
cfg_path = "D:\Fragss-AI-main\Fragss-AI-main\yolov3.cfg"
weights_path = "D:\Fragss-AI-main\Fragss-AI-main\yolo.weights"

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_PIXEL_VALUE = 255
NO_OF_CHANNELS = 3
TIMESTEPS = 10
VIDEO_FOLDER_PATH = "output_segments/videos"
VIRALITY_FOLDER_PATH = os.path.join(VIDEO_FOLDER_PATH, "clip_virality")
os.makedirs(VIRALITY_FOLDER_PATH, exist_ok=True)
YOLO_WEIGHTS = 'yolov3.weights'
YOLO_CFG = 'yolov3.cfg'

# Mapping from class id to event name
CLASS_ID_TO_EVENT = {
    0: "multiple_kills", 1: "headshots", 2: "clutch_round", 3: "triple_kill",
    4: "quadra_kill", 5: "ace", 6: "explosions", 7: "ultimate_ability_used",
    8: "smoke_screen", 9: "flank_kill", 10: "defuse_bomb", 11: "plant_bomb",
    12: "sniper_kill", 13: "ability_kill", 14: "smoke_bait", 15: "retake_win",
    16: "flawless_victory", 17: "strategic_baiting", 18: "team_kill", 19: "special_ability_ultimate",
    20: "killstreak_5", 21: "final_blow"
}

# Event Weights
EVENT_WEIGHTS = {
    "multiple_kills": 3.5, "headshots": 3.5, "clutch_round": 6.0, "triple_kill": 4.0,
    "quadra_kill": 4.5, "ace": 6.0, "explosions": 2.8, "ultimate_ability_used": 2.5,
    "smoke_screen": 1.3, "flank_kill": 3.2, "defuse_bomb": 3.0, "plant_bomb": 2.5,
    "sniper_kill": 3.8, "ability_kill": 3.0, "smoke_bait": 2.0, "retake_win": 5.0,
    "flawless_victory": 5.0, "strategic_baiting": 2.0, "team_kill": -1.0,
    "special_ability_ultimate": 2.5, "killstreak_5": 5.0, "final_blow": 3.3
}

# Load action detection model
def load_action_model(model_path):
    logging.info(f"Loading action detection model from: {model_path}")
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e
    return model

# Extract frames
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

    while len(frames_list) < TIMESTEPS:
        frames_list.append(frames_list[-1])

    return frames_list[:TIMESTEPS]

# Sentiment Analysis
def sentiment_analysis(text):
    vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer.polarity_scores(text)["compound"]

def predict_actions(video_path, weights_path, cfg_path):
    detections = extract_features(video_path, weights_path, cfg_path, frame_rate=5)
    logging.info(f"Detections for {video_path}: {detections}")

    event_count = {}

    if not detections:
        logging.warning(f"No detections found for {video_path}")
        return 0.5  # fallback

    for det in detections:
        event = det['action']
        event_count[event] = event_count.get(event, 0) + 1

    weighted_action_score = sum(EVENT_WEIGHTS.get(event, 0.5) * count for event, count in event_count.items())
    return max(weighted_action_score, 0.1)

# Compute Virality Score
def compute_virality_score(sentiment_score, weighted_action_score):
    sentiment_weight = max(0.2, min(0.5, abs(sentiment_score) / 2))
    action_weight = 0.9
    random_variance = np.random.uniform(1.1, 1.3)

    virality_score = ((sentiment_weight * sentiment_score) + (action_weight * weighted_action_score)) * random_variance
    virality_score = np.clip(virality_score, 1.0, 10.0)
    return round(virality_score, 2)

# Predict Virality
def predict_virality(video_path, config=None):
    sentiment_score = 0  # Optional, skip for now
    weighted_action_score = predict_actions(video_path, weights_path, cfg_path)

    logging.info(f"Weighted action score for {video_path}: {weighted_action_score}")

    sentiment_weight = 0.1
    action_weight = 0.9
    random_variance = np.random.uniform(1.1, 1.3)

    virality_score = ((sentiment_weight * sentiment_score) + (action_weight * weighted_action_score)) * random_variance
    virality_score = np.clip(virality_score, 1.0, 10.0)

    logging.info(f"Virality score for {video_path}: {virality_score:.2f}")
    return round(virality_score, 2)


# Rank Clips
def rank_clips(video_paths, model_args):
    weights_path, cfg_path = model_args
    ranked = []
    for path in video_paths:
        score = predict_virality(path, config={"weights_path": weights_path, "cfg_path": cfg_path})
        ranked.append((path, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

import shutil

def copy_files(ranked_clip_paths, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for clip_tuple in ranked_clip_paths:
        # If it's a tuple, extract the path from the first element
        clip_path = clip_tuple[0] if isinstance(clip_tuple, tuple) else clip_tuple

        if isinstance(clip_path, (str, bytes, os.PathLike)) and os.path.isfile(clip_path):
            filename = os.path.basename(clip_path)
            dest_path = os.path.join(destination_folder, filename)
            shutil.copy2(clip_path, dest_path)
            logging.info(f"Copied: {clip_path} → {dest_path}")
        else:
            logging.warning(f"Clip not found or invalid path: {clip_path}")


# Get video clips from folder
def get_video_clips_from_folder(folder_path):
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(valid_extensions)]
