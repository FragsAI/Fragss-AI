import os
import numpy as np
import cv2
import logging
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_yolo3.yolo import YOLO
from keras_yolo3.utils import preprocess_image
from keras.utils import to_categorical

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("action_detection.log"),
    logging.StreamHandler()
])

# Create the YOLO model
def initialize_yolo_model():
    logging.info("Initializing YOLO model.")
    return YOLO(
        model_path='model_data/yolo.h5',
        anchors_path='model_data/yolo_anchors.txt',
        classes_path='model_data/coco_classes.txt',
        score=0.3,
        iou=0.5,
        model_image_size=(416, 416)
    )

# Load and preprocess videos for YOLO
def load_and_preprocess_video(video_path, frame_size=(416, 416)):
    logging.info(f"Loading and preprocessing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = preprocess_image(frame, frame_size)
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Extract features from a video using YOLO
def extract_video_features(video_frames, yolo):
    logging.info("Extracting features from video frames using YOLO.")
    features = []
    for frame in video_frames:
        image_data = frame / 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension
        yolo_outputs = yolo.model.predict(image_data)
        features.append(yolo_outputs)
    avg_features = np.mean(features, axis=0)
    return avg_features

# Load videos from folder and create dataset
def generate_dataset(folder, label, frame_size=(416, 416), chunk_size=16, overlap=8):
    logging.info(f"Generating dataset from folder: {folder}")
    videos, labels = [], []
    for filename in os.listdir(folder):
        video_path = os.path.join(folder, filename)
        if not video_path.endswith(('.mp4', '.avi')):
            continue
        frames = load_and_preprocess_video(video_path, frame_size)
        
        # Divide video into chunks
        num_chunks = (len(frames) - chunk_size) // overlap + 1
        for i in range(num_chunks):
            chunk = frames[i * overlap : i * overlap + chunk_size]
            if len(chunk) == chunk_size:
                videos.append(chunk)
                labels.append(label)
    return np.array(videos), np.array(labels)

def create_training_data():
    logging.info("Creating training dataset.")
    training_folder = './training_videos/'
    x_train, y_train = generate_dataset(training_folder, label=0)
    y_train = to_categorical(y_train, num_classes=2)
    return x_train, y_train

# Create a simple dense model for action classification
def build_action_classification_model(input_shape, num_classes):
    logging.info("Building action classification model.")
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Train the action detection model with model saving
def train_action_classification_model(x_train, y_train, x_val, y_val, input_shape, num_classes, model_path='action_model.h5'):
    logging.info("Training action classification model.")
    action_model = build_action_classification_model(input_shape, num_classes)
    action_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    action_model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping, model_checkpoint])
    return action_model

# Load a trained action model
def load_trained_action_model(model_path='action_model.h5'):
    if os.path.exists(model_path):
        logging.info(f"Loading trained action model from {model_path}.")
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"No saved model found at {model_path}")

# Classify a video using the trained action model
def predict_video_class(video_path, yolo, action_model, threshold=0.8):
    logging.info(f"Classifying video: {video_path}")
    video_frames = load_and_preprocess_video(video_path)
    avg_features = extract_video_features(video_frames, yolo)
    norm_features = avg_features / np.linalg.norm(avg_features)
    
    predictions = action_model.predict(np.expand_dims(norm_features, axis=0))
    labels = ['training']
    predicted_labels = [labels[i] for i, confidence in enumerate(predictions[0]) if confidence > threshold]

    logging.info(f"Predicted labels for the video: {predicted_labels if predicted_labels else ['other']}")
    return predicted_labels if predicted_labels else ['other']

# Main script
if __name__ == "__main__":
    yolo = initialize_yolo_model()
    model_path = 'action_model.h5'
    
    try:
        action_model = load_trained_action_model(model_path)
        logging.info("Loaded existing action model.")
    except FileNotFoundError:
        logging.info("Training new action model.")
        x_train, y_train = create_training_data()

        # Split the dataset into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        # Train the action detection model
        input_shape = (13,)  # Adjust based on the YOLO output shape
        num_classes = 2
        action_model = train_action_classification_model(x_train, y_train, x_val, y_val, input_shape, num_classes, model_path)

    # Example usage
    video_path = 'example_video.mp4'
    predicted_class = predict_video_class(video_path, yolo, action_model, threshold=0.8)
    logging.info(f'Predicted class for the video: {predicted_class}')

