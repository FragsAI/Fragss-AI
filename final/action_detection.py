import os
import cv2
import numpy as np
import logging
import datetime as dt
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
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
NO_OF_CLASSES = len(CLASS_LABELS)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("action_detection.log"), logging.StreamHandler()])

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
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Extract features from frames using YOLO
def extract_features(video_path, frame_rate=5):
    net, output_layers = load_yolo_model()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features = []

    logging.info(f"Extracting features from {video_path}")
    for frame_idx in tqdm(range(0, total_frames, frame_rate), desc="Extracting frames"):
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
                    features.append(detection)

    cap.release()
    return np.array(features)

# Create ConvLSTM2D model
def create_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                         input_shape=(None, 13, 13, 1)))
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(NO_OF_CLASSES, activation='softmax'))
    return model

# Train and evaluate the model
def train_and_evaluate_model(features_train, labels_train, features_test, labels_test):
    model = create_model()
    early_stopping_callback = EarlyStopping(patience=7, restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    logging.info("Starting model training")
    model_training_history = model.fit(
        features_train, labels_train,
        epochs=50,
        batch_size=64,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback]
    )

    logging.info("Evaluating the model")
    model_evaluation_history = model.evaluate(features_test, labels_test)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

    logging.info(f"Test Loss: {model_evaluation_loss}, Test Accuracy: {model_evaluation_accuracy}")

    # Save the model
    logging.info("Saving the model")
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss:.4f}___Accuracy_{model_evaluation_accuracy:.4f}.h5'
    model.save(model_name)

    # Print model summary after saving
    model.summary()

    # Predictions and classification report
    labels_pred = model.predict(features_test)
    labels_pred = np.argmax(labels_pred, axis=1)
    labels_test = np.argmax(labels_test, axis=1)

    logging.info("Classification Report:")
    print(classification_report(labels_test, labels_pred, target_names=CLASS_LABELS))

if __name__ == "__main__":
    video_path = 'your_video_path_here.mp4'
    features = extract_features(video_path)
    # Assuming labels_train, labels_test, features_train, features_test are already prepared.
    # Replace with your own training/testing dataset code.
    train_and_evaluate_model(features_train, labels_train, features_test, labels_test)
