import os
import cv2
import numpy as np

# Constants
YOLO_WEIGHTS = 'yolov3.weights'
YOLO_CFG = 'yolov3.cfg'
CONFIDENCE_THRESHOLD = 0.6  # Stricter threshold to reduce false positives
NMS_THRESHOLD = 0.3        # Stricter NMS threshold
INPUT_SIZE = (416, 416)

# Custom class labels for gaming actions
CLASS_LABELS = [
    "gunshot",       # Shooting action
    "grenade_throw", # Throwing a grenade
    "knife_attack",  # Melee attack with a knife
    "multiple_kills",# Multiple kills in quick succession
    "reload",        # Reloading weapon
    "headshot",      # Headshot
    "sniper_shot",   # Sniper rifle shot
    "pistol_shot",   # Pistol shot
    "explosion",     # Explosion event
    "death",         # Player death
    "heal",          # Healing action
    "revive",        # Reviving teammate
    "crouch",        # Crouching
    "jump",          # Jumping action
    "sprint",        # Sprinting
    "capture_flag",  # Capturing objective or flag
    "use_medkit",    # Using a medkit or health pack
    "use_shield",    # Using a shield or protective gear
    "taunt",         # Taunting
    "pickup_item"    # Picking up an item
]

# Download YOLOv3 weights and config if not present
def download_yolo_files():
    if not os.path.exists(YOLO_WEIGHTS):
        os.system(f'wget https://pjreddie.com/media/files/yolov3.weights')
    if not os.path.exists(YOLO_CFG):
        os.system(f'wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')

# Load the YOLO model
def load_yolo_model():
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:  # Fallback for older OpenCV versions
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Detect actions in a frame
def detect_actions(frame, net, output_layers):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, INPUT_SIZE, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

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

                # Ignore detections in irrelevant regions (e.g., very small objects)
                if w < 50 or h < 50:  # Example threshold for minimum size
                    continue

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    detections = [(class_ids[i], boxes[i], confidences[i]) for i in indices]  # No flatten()

    return detections

# Process the video for action detection
def process_video(video_path, net, output_layers, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_actions(frame, net, output_layers)

        for class_id, box, confidence in detections:
            x, y, w, h = box
            label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else str(class_id)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)  # Save frame to video
        cv2.imshow("Action Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video processing interrupted by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Download YOLO files if not already downloaded
    download_yolo_files()

    # Load the YOLO model
    net, output_layers = load_yolo_model()

    # Path to the input and output video
    video_path = '/Users/levent/Documents/FragsAI/Video/videoplayback (1).mp4'
    output_path = '/Users/levent/Documents/FragsAI/Video/video_output3.mp4'
    process_video(video_path, net, output_layers, output_path)