import os
import numpy as np
import cv2
import logging
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_yolo3.yolo import YOLO
from keras_yolo3.utils import preprocess_image
from keras.utils import to_categorical

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### Preprocessing Functions ###

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video and saves them as JPEG files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def preprocess_frames(input_folder, output_file, resize_dim=(224, 224), augment=False):
    """
    Preprocesses extracted frames by resizing and normalizing them, and saves as a numpy array.
    """
    def augment_frame(frame):
        """
        Apply random transformations to augment the frame.
        """
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        if np.random.rand() > 0.5:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        if np.random.rand() > 0.5:
            frame = np.array(255 * frame, dtype='uint8')
        return frame

    frames = []
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".jpg"):
            frame_path = os.path.join(input_folder, file_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame = cv2.resize(frame, resize_dim)
            frame = frame.astype(np.float32) / 255.0
            if augment:
                frame = augment_frame(frame)
            frames.append(frame)
    
    frames = np.array(frames)
    np.save(output_file, frames)

### Shot Boundary Detection Functions ###

def extract_frames_multithreaded(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}

    for frame_index in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[frame_index] = gray

    cap.release()
    return frames

def calculate_histogram_difference(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def detect_shot_boundaries(video_path, method='sift', diff_threshold=50000, match_threshold=0.7, hist_diff_threshold=0.5, num_threads=4):
    """
    Detects shot boundaries in a video using SIFT, frame differencing, or histogram differences.
    """
    frames = extract_frames_multithreaded(video_path)
    if method == 'sift':
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    prev_frame = None
    prev_des = None
    shot_boundaries = []
    
    logging.info(f"Total frames in video: {len(frames)}")
    
    with tqdm(total=len(frames), desc="Detecting shot boundaries") as pbar:
        for frame_index in sorted(frames.keys()):
            gray = frames[frame_index]
            
            if method == 'diff':
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    non_zero_count = np.count_nonzero(diff)
                    
                    if non_zero_count > diff_threshold:
                        shot_boundaries.append(frame_index)
                prev_frame = gray
            
            elif method == 'sift':
                kp, des = sift.detectAndCompute(gray, None)
                
                if prev_des is not None:
                    matches = bf.match(prev_des, des)
                    distances = [m.distance for m in matches]
                    
                    if len(distances) > 0:
                        avg_distance = sum(distances) / len(distances)
                        if avg_distance > match_threshold:
                            shot_boundaries.append(frame_index)
                prev_des = des
            
            elif method == 'hist':
                if prev_frame is not None:
                    hist_diff = calculate_histogram_difference(prev_frame, gray)
                    if hist_diff < hist_diff_threshold:
                        shot_boundaries.append(frame_index)
                prev_frame = gray
            
            pbar.update(1)
    
    logging.info(f"Detected {len(shot_boundaries)} shot boundaries.")
    return shot_boundaries

### Clip Segmentation Functions ###

def segment_clips(video_path, shot_boundaries, output_folder):
    """
    Segments a video into clips based on detected shot boundaries.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    shot_boundaries.append(total_frames)  # Ensure the last segment includes until the end of the video
    
    for i in range(len(shot_boundaries) - 1):
        start_frame = shot_boundaries[i]
        end_frame = shot_boundaries[i + 1]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_frames = []
        
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            clip_frames.append(frame)
        
        output_clip_path = os.path.join(output_folder, f"clip_{i:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_clip_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        
        for frame in clip_frames:
            out.write(frame)
        
        out.release()
    
    cap.release()
    cv2.destroyAllWindows()

### Virality Ranking Functions ###

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

### Action Detection Functions ###

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

def extract_video_features_with_yolo(video_frames, yolo):
    logging.info("Extracting features from video frames using YOLO.")
    features = []
    for frame in video_frames:
        image_data = frame / 255.0
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension
        yolo_outputs = yolo.model.predict(image_data)
        features.append(yolo_outputs)
    avg_features = np.mean(features, axis=0)
    return avg_features

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

def build_action_classification_model(input_shape, num_classes):
    logging.info("Building action classification model.")
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_action_classification_model(x_train, y_train, x_val, y_val, input_shape, num_classes, model_path='action_model.h5'):
    logging.info("Training action classification model.")
    action_model = build_action_classification_model(input_shape, num_classes)
    action_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    action_model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping, model_checkpoint])
    return action_model

def load_trained_action_model(model_path='action_model.h5'):
    if os.path.exists(model_path):
        logging.info(f"Loading trained action model from {model_path}.")
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"No saved model found at {model_path}")

def predict_video_class(video_path, yolo, action_model, threshold=0.8):
    logging.info(f"Classifying video: {video_path}")
    video_frames = load_and_preprocess_video(video_path)
    avg_features = extract_video_features_with_yolo(video_frames, yolo)
    norm_features = avg_features / np.linalg.norm(avg_features)
    
    predictions = action_model.predict(np.expand_dims(norm_features, axis=0))
    labels =[]
    predicted_labels = [labels[i] for i, confidence in enumerate(predictions[0]) if confidence > threshold]

    logging.info(f"Predicted labels for the video: {predicted_labels if predicted_labels else ['other']}")
    return predicted_labels if predicted_labels else ['other']

### Main Pipeline Function ###

def main_pipeline(video_path, output_folder, model_file, scaler_file):
    # Preprocessing
    logging.info("Starting preprocessing")
    frames_folder = "frames"
    extract_frames(video_path, frames_folder, frame_rate=5)
    preprocess_frames(frames_folder, 'preprocessed_frames.npy', resize_dim=(224, 224), augment=True)
    
    # Shot Boundary Detection
    logging.info("Starting shot boundary detection")
    shot_boundaries = detect_shot_boundaries(video_path, method='sift')
    
    # Clip Segmentation
    logging.info("Starting clip segmentation")
    segment_clips(video_path, shot_boundaries, output_folder)
    
    # Virality Ranking
    logging.info("Ranking clips based on virality")
    clip_paths = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.mp4')]
    model, scaler = load_model(model_file, scaler_file)
    ranked_clips = rank_clips(clip_paths, model, scaler)
    
    logging.info("Ranked clips:")
    for score, clip_path in ranked_clips:
        logging.info(f"Clip: {clip_path}, Score: {score}")
    
    # Action Detection
    logging.info("Starting action detection")
    yolo = initialize_yolo_model()
    action_model_path = 'action_model.h5'
    
    try:
        action_model = load_trained_action_model(action_model_path)
        logging.info("Loaded existing action model.")
    except FileNotFoundError:
        logging.info("Training new action model.")
        x_train, y_train = create_training_data()

        # Split the dataset into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        # Train the action detection model
        input_shape = (13,)  # Adjust based on the YOLO output shape
        num_classes = 2
        action_model = train_action_classification_model(x_train, y_train, x_val, y_val, input_shape, num_classes, action_model_path)
    
    # Example usage of action detection
    for clip_path in clip_paths:
        predicted_class = predict_video_class(clip_path, yolo, action_model, threshold=0.8)
        logging.info(f'Predicted class for clip {clip_path}: {predicted_class}')

if __name__ == "__main__":
    video_path = "example_video.mp4"
    output_folder = "clips"
    model_file = 'virality_model.pkl'
    scaler_file = 'scaler.pkl'
    
    main_pipeline(video_path, output_folder, model_file, scaler_file)
