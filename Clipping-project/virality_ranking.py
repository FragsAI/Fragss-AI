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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_PIXEL_VALUE = 255
NO_OF_CHANNELS = 3
TIMESTEPS = 10
VIDEO_FOLDER_PATH = "output_segments/videos"
VIRALITY_FOLDER_PATH = os.path.join(VIDEO_FOLDER_PATH, "clip_virality")
os.makedirs(VIRALITY_FOLDER_PATH, exist_ok=True)

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

# Predict Actions
def predict_actions(frames, model, action_threshold=0.1):
    features = np.array(frames).reshape(1, TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    predictions = model.predict(features).flatten()
    action_probabilities = softmax(predictions * 5)
    
    weighted_action_score = 0
    for i, prob in enumerate(action_probabilities):
        if prob > action_threshold:
            event_name = CLASS_ID_TO_EVENT.get(i, None)
            if event_name:
                weighted_action_score += prob * EVENT_WEIGHTS.get(event_name, 0.5)
    
    return max(weighted_action_score * 10, 0.1)

# Compute Virality Score
def compute_virality_score(sentiment_score, weighted_action_score):
    sentiment_weight = max(0.2, min(0.5, abs(sentiment_score) / 2))
    action_weight = 0.9
    random_variance = np.random.uniform(1.1, 1.3)

    virality_score = ((sentiment_weight * sentiment_score) + (action_weight * weighted_action_score)) * random_variance
    virality_score = np.clip(virality_score, 1.0, 10.0)
    return round(virality_score, 2)

# Predict Virality
def predict_virality(video_path, action_model):
    frames = extract_frames(video_path)
    sentiment_score = 0
    weighted_action_score = predict_actions(frames, action_model)
    virality_score = compute_virality_score(sentiment_score, weighted_action_score)
    logging.info(f"Predicted virality score for {video_path}: {virality_score}")
    return virality_score

# Rank Clips
def rank_clips(video_clips, action_model):
    ranked_clips = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(predict_virality, clip, action_model): clip for clip in video_clips}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Ranking clips"):
            clip = futures[future]
            try:
                score = future.result()
                ranked_clips.append((score, clip))
            except Exception as e:
                logging.error(f"Error processing video {clip}: {e}")

    ranked_clips.sort(reverse=True, key=lambda x: x[0])
    return ranked_clips

# Copy Files
def copy_files(ranked_clips):
    for rank, (score, clip) in enumerate(ranked_clips, start=1):
        folder, filename = os.path.split(clip)
        name, ext = os.path.splitext(filename)
        new_name = f"{rank:02d}_{name}_score_{score:.2f}{ext}"
        new_path = os.path.join(VIRALITY_FOLDER_PATH, new_name)

        command = ["ffmpeg", "-i", clip, "-c:v", "copy", "-c:a", "copy", new_path, "-y"]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logging.info(f"Copied {clip} -> {new_path}")

# Get video clips from folder
def get_video_clips_from_folder(folder_path):
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(valid_extensions)]
