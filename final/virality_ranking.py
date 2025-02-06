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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_PIXEL_VALUE = 255
NO_OF_CHANNELS = 3
TIMESTEPS = 10
VIDEO_FOLDER_PATH = "output_segments/videos"
VIRALITY_FOLDER_PATH = os.path.join(VIDEO_FOLDER_PATH, "clip_virality")
os.makedirs(VIRALITY_FOLDER_PATH, exist_ok=True)

# Define event intensity weights (Higher scale for intense events)
EVENT_WEIGHTS = {
    "multiple_kills": 3.5,          # More weight for major action (multiple kills in quick succession)
    "headshots": 3.5,               # Headshots, especially with precise aiming (high skill moment)
    "clutch_round": 6.0,            # Clutch moments where a player wins the round alone (highest impact)
    "triple_kill": 4.0,             # Triple kill in a round (high skill or luck moment)
    "quadra_kill": 4.5,             # Quadra kill (exceptional play)
    "ace": 6.0,                     # Ace (killing all enemies in a round by yourself)
    "explosions": 2.8,              # Explosions, such as kills with grenades (impactful but not as much as gunplay)
    "ultimate_ability_used": 2.5,   # Using an ultimate ability (game-changing plays)
    "smoke_screen": 1.3,            # Deploying smokes or blocking vision (strategic, but not immediately impactful)
    "flank_kill": 3.2,              # Getting a kill while flanking an enemy (strategic and impactful)
    "defuse_bomb": 3.0,             # Defusing the bomb, particularly under pressure
    "plant_bomb": 2.5,              # Planting the bomb, signaling a critical moment
    "sniper_kill": 3.8,             # Sniper kill (long-range precision, skillful play)
    "ability_kill": 3.0,            # Killing an enemy using an agent’s ability (highly impactful when used well)
    "smoke_bait": 2.0,              # Using smokes to bait enemies into traps (tactical play)
    "retake_win": 5.0,              # Winning a retake (when the defending team loses control but takes it back)
    "flawless_victory": 5.0,        # Winning the round without losing a single player (significant momentum)
    "strategic_baiting": 2.0,       # Tactical baiting where the enemy is tricked by fake plays
    "team_kill": -1.0,              # Team kills (negative impact, penalize this)
    "special_ability_ultimate": 2.5,# Using special or game-changing ultimates (e.g., Phoenix’s resurrection, Omen's teleport)
    "killstreak_5": 5.0,            # Killing streak (5 or more kills in a row without dying)
    "final_blow": 3.3,              # Delivering the final blow in crucial rounds (clutch moment)
}

# Load action detection model
def load_action_model(model_path):
    logging.info(f"Loading action detection model from: {model_path}")
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        logging.error(f"Error loading action detection model: {e}")
        raise e
    return model

# Extract frames from video
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

    # Pad with last frame if needed
    while len(frames_list) < TIMESTEPS:
        frames_list.append(frames_list[-1])

    return frames_list[:TIMESTEPS]

# Perform sentiment analysis on the video transcript (if available)
def sentiment_analysis(text):
    vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer.polarity_scores(text)["compound"]

# Predict actions in a video and return score, ignoring low probability events
def predict_actions(frames, model, action_threshold=0.1):
    features = np.array(frames).reshape(1, TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    predictions = model.predict(features).flatten()
    action_probabilities = softmax(predictions * 5)  
    action_probabilities = [p if p > action_threshold else 0 for p in action_probabilities]
    weighted_action_score = sum(
        action_probabilities[i] * EVENT_WEIGHTS.get(f"action_{i}", 0.5) for i in range(len(action_probabilities))
    )
    if weighted_action_score == 0:
        return 0.1  

    return weighted_action_score * 10 

# Compute final virality score with dynamic weighting and keep the value between 1 and 10
def compute_virality_score(sentiment_score, weighted_action_score):
    sentiment_magnitude = abs(sentiment_score)
    sentiment_weight = max(0.2, min(0.5, sentiment_magnitude / 2)) 

    action_weight = 0.9
    
    random_variance = np.random.uniform(1.5, 2.5)  # More aggressive variation


    virality_score = ((sentiment_weight * sentiment_score) + (action_weight * weighted_action_score)) * random_variance
    
    virality_score = np.clip(virality_score, 1.00, 10.00)  # Clip the score between 1 and 10
    return round(virality_score, 2) 

# Predict virality score for a given video
def predict_virality(video_path, action_model):
    frames = extract_frames(video_path)
    
    sentiment_score = 0  
    
    weighted_action_score = predict_actions(frames, action_model)

    virality_score = compute_virality_score(sentiment_score, weighted_action_score)
    logging.info(f"Predicted virality score for {video_path}: {virality_score}")
    
    return virality_score

# Rank videos based on virality score
def rank_clips(video_clips, action_model):
    ranked_clips = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(predict_virality, clip, action_model): clip
            for clip in video_clips
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing clips"):
            clip = futures[future]
            try:
                score = future.result()
                ranked_clips.append((score, clip))
            except Exception as e:
                logging.error(f"Error processing video {clip}: {e}")

    # Sorting the clips from highest to lowest virality score
    ranked_clips.sort(reverse=True, key=lambda x: x[0])
    return ranked_clips

# Copy ranked videos to clip_virality folder while keeping audio
def copy_files(ranked_clips):
    for rank, (score, clip) in enumerate(sorted(ranked_clips, reverse=True), start=1):
        folder, filename = os.path.split(clip)
        name, ext = os.path.splitext(filename)
        new_name = f"{rank:02d}_{name}_score_{score:.2f}{ext}"  
        new_path = os.path.join(VIRALITY_FOLDER_PATH, new_name)

        # Use FFmpeg to ensure audio is preserved
        command = [
            "ffmpeg", "-i", clip, "-c:v", "copy", "-c:a", "copy", new_path, "-y"
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logging.info(f"Copied {clip} -> {new_path} with audio intact")


# Get all video clips in `output_segments/`
def get_video_clips_from_folder(folder_path):
    valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(valid_extensions)]

# if __name__ == "__main__":
#     MODEL_PATH = ("Enter model path")  # Path to action detection model
#     action_model = load_action_model(MODEL_PATH)
    
#     video_clips = get_video_clips_from_folder(VIDEO_FOLDER_PATH)
    
#     if not video_clips:
#         logging.info("No video clips found in the folder.")
#     else:
#         ranked_clips = rank_clips(video_clips, action_model)
#         copy_files(ranked_clips)
#         logging.info("Processing complete! Files copied to 'clip_virality'.")
