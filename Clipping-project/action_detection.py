import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_CLASS_LABELS = [
    "gunshot", "grenade_throw", "knife_attack", "multiple_kills", "reload",
    "headshot", "sniper_shot", "pistol_shot", "explosion", "death",
    "heal", "revive", "crouch", "jump", "sprint",
    "capture_flag", "use_medkit", "use_shield", "taunt", "pickup_item"
]

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = (416, 416)

# Load YOLO model
def load_yolo_model(weights_path, cfg_path, class_labels=DEFAULT_CLASS_LABELS):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers, class_labels

# Extract features from video
def extract_features(video_path, weights_path, cfg_path, frame_rate=5, class_labels=DEFAULT_CLASS_LABELS):
    net, output_layers, labels = load_yolo_model(weights_path, cfg_path, class_labels)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    actions_detected = []

    logging.info(f"Extracting features from {video_path}, total frames: {total_frames}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
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
                            'action': labels[class_id],
                            'confidence': float(confidence),
                            'box': [x, y, w, h]
                        })

        frame_idx += 1

    cap.release()

    if not actions_detected:
        logging.warning("No actions detected in the video.")

    return actions_detected
