import cv2
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime as dt
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
TIMESTEPS = 10 
CLASS_CATEGORIES_LIST = ["Nunchucks", "Punch"]
MAX_PIXEL_VALUE = 255
BATCH_SIZE = 100
NO_OF_CHANNELS = 3
NO_OF_CLASSES = len(CLASS_CATEGORIES_LIST)

# Function to extract frames from a video
def extract_frames(video_path):
    frames_list = []
    videoObj = cv2.VideoCapture(video_path)
    logging.info(f"Extracting frames from video: {video_path}")
    while True:
        success, image = videoObj.read()
        if not success:
            break
        resized_frame = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / MAX_PIXEL_VALUE
        frames_list.append(normalized_frame)
    videoObj.release()
    while len(frames_list) < TIMESTEPS:
        frames_list.append(frames_list[-1])
    return np.array(frames_list[:TIMESTEPS])

# Function to process a batch of videos
def process_batch(class_dir, class_index, video_files):
    features = []
    labels = []
    logging.info(f"Processing batch for class index: {class_index}")
    for video_name in video_files:
        video_path = os.path.join(class_dir, video_name)
        frames = extract_frames(video_path)
        features.append(frames)
        labels.append(class_index)
    return np.array(features), np.array(labels)

# Prepare datasets
dataset_dir = '/Users/kesinishivaram/FragsAI/UCF50'  # Adjust path to your dataset
all_features = []
all_labels = []

logging.info("Preparing datasets")
for class_index, class_name in enumerate(CLASS_CATEGORIES_LIST):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        raise ValueError(f"Directory {class_dir} does not exist")
    video_files = os.listdir(class_dir)
    num_batches = len(video_files) // BATCH_SIZE + (1 if len(video_files) % BATCH_SIZE != 0 else 0)
    for batch_num in range(num_batches):
        batch_files = video_files[batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE]
        batch_features, batch_labels = process_batch(class_dir, class_index, batch_files)
        all_features.append(batch_features)
        all_labels.append(batch_labels)

# Concatenate all features and labels
logging.info("Concatenating all features and labels")
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Convert labels to one-hot encoding
logging.info("Converting labels to one-hot encoding")
all_labels = to_categorical(all_labels, num_classes=NO_OF_CLASSES)

# Split data into training and test sets
logging.info("Splitting data into training and test sets")
features_train, features_test, labels_train, labels_test = train_test_split(
    all_features, all_labels, test_size=0.2, shuffle=True, random_state=0
)

# Reshape features for ConvLSTM2D input
features_train = features_train.reshape(features_train.shape[0], TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
features_test = features_test.reshape(features_test.shape[0], TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)

# Define the model
def create_model():
    model = Sequential()
    input_shape = (TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    model.add(Input(shape=input_shape))  # Use Input layer to specify the input shape
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(NO_OF_CLASSES, activation='softmax'))
    return model

model = create_model()

# Compile the model
logging.info("Compiling the model")
early_stopping_callback = EarlyStopping(patience=7, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
logging.info("Starting model training")
model_training_history = model.fit(
    features_train, labels_train, 
    epochs=2, 
    batch_size=64, 
    shuffle=True, 
    validation_split=0.2, 
    callbacks=[early_stopping_callback]
)

# Evaluate the model
logging.info("Evaluating the model")
model_evaluation_history = model.evaluate(features_test, labels_test)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Save the model
logging.info("Saving the model")
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'
model.save(model_name)

# Print model summary after saving
logging.info("Model Summary after saving:")
model.summary()

# Predictions
logging.info("Making predictions on the test set")
labels_pred = model.predict(features_test)
labels_pred = np.argmax(labels_pred, axis=1)
labels_test = np.argmax(labels_test, axis=1)

# Print evaluation results
logging.info(f"Test Loss: {model_evaluation_loss}, Test Accuracy: {model_evaluation_accuracy}")

# Print classification report
logging.info("Classification Report:")
print(classification_report(labels_test, labels_pred, target_names=CLASS_CATEGORIES_LIST))
