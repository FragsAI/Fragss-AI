# action_detection.py

import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Constants
YOLO_WEIGHTS = r'C:\Users\mdama\Downloads\newww\Fragss-AI-main\final\yolo.weights'
YOLO_CFG = r'C:\Users\mdama\Downloads\newww\Fragss-AI-main\final\yolov3.cfg'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = (416, 416)
CLASS_LABELS = [
    "gunshot", "grenade_throw", "knife_attack", "multiple_kills", "reload",
    "headshot", "sniper_shot", "pistol_shot", "explosion", "death",
    "heal", "revive", "crouch", "jump", "sprint",
    "capture_flag", "use_medkit", "use_shield", "taunt", "pickup_item"
]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def extract_features(video_path, frame_rate=5):
    net, output_layers = load_yolo_model()
    
    if not output_layers:
        logging.error("Error: YOLO model failed to load correctly. Check model files.")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actions_detected = []

    logging.info(f"Extracting features from {video_path}, total frames: {total_frames}")

    for frame_idx in tqdm(range(0, total_frames, frame_rate), desc="Detecting actions"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Warning: Unable to read frame {frame_idx}, skipping.")
            continue

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Use a threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    actions_detected.append({
                        'frame': frame_idx,
                        'action': CLASS_LABELS[class_id],
                        'confidence': float(confidence),
                        'box': [x, y, w, h]
                    })
    
    cap.release()
    
    if not actions_detected:
        logging.warning("No actions detected in the video.")

    return actions_detected
