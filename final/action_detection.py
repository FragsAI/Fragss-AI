# action_detection.py

import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Constants
YOLO_WEIGHTS = 'yolov3.weights'
YOLO_CFG = 'yolov3.cfg'
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

# Ensure YOLO files are available
def download_yolo_files():
    if not os.path.exists(YOLO_WEIGHTS):
        os.system(f'wget https://pjreddie.com/media/files/yolov3.weights')
    if not os.path.exists(YOLO_CFG):
        os.system(f'wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')

# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Extract features (action detections) from frames using YOLO
def extract_features(video_path, frame_rate=5):
    """
    Detects actions in video frames using YOLO.
    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Interval for frame extraction (every nth frame).
    Returns:
        list: Detected actions and their details (frame, action type, confidence, box).
    """
    net, output_layers = load_yolo_model()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actions_detected = []

    logging.info(f"Extracting features from {video_path}")
    for frame_idx in tqdm(range(0, total_frames, frame_rate), desc="Detecting actions"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, INPUT_SIZE, (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
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
    return actions_detected
