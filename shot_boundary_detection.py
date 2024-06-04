import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def preprocess_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

def detect_edges(frame):
    # Apply Canny edge detection
    edges = cv2.Canny(frame, threshold1=50, threshold2=150)
    return edges

def calculate_histogram(frame):
    # Calculate histogram of the grayscale frame
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    return hist.flatten()

def calculate_histogram_difference(hist1, hist2):
    # Compare two histograms using Bhattacharyya distance
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return hist_diff

def extract_sift_features(frame):
    # Extract SIFT features from the frame
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    return keypoints, descriptors

def extract_surf_features(frame):
    # Extract SURF features from the frame
    surf = cv2.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(frame, None)
    return keypoints, descriptors

def extract_hog_features(frame):
    # Extract HOG features from the frame
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(frame)
    return hog_features

def train_svm_model(X_train, y_train):
    # Train SVM model using scaled feature vectors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svm = SVC(kernel='linear')
    svm.fit(X_train_scaled, y_train)
    return svm

def classify_shot_boundaries(frames, labels, svm_model):
    # Classify shot boundaries using SVM model
    feature_vectors = []
    for frame in frames:
        # Example: Extract SIFT features
        keypoints, descriptors = extract_sift_features(frame)
        if descriptors is not None:
            feature_vectors.append(descriptors.flatten())

    # Scale feature vectors
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)

    # Predict shot boundaries using SVM
    predictions = svm_model.predict(scaled_features)
    shot_boundaries = [i for i, label in enumerate(predictions) if label == 1]
    return shot_boundaries

# Baseline shot boundary detection using edge detection and histogram-based methods
def shot_boundary_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return []

    prev_frame = None
    shot_boundaries = [0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        gray_frame = preprocess_frame(frame)

        # Edge detection
        edges = detect_edges(gray_frame)

        # Histogram calculation
        hist = calculate_histogram(gray_frame)
        if prev_frame is not None:
            prev_hist = calculate_histogram(prev_frame)
            hist_diff = calculate_histogram_difference(prev_hist, hist)
            if hist_diff > 0.5:  # Adjust the threshold as needed
                shot_boundaries.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        prev_frame = gray_frame

    cap.release()
    return shot_boundaries

# Shot boundary detection using machine learning with feature extraction
def shot_boundary_detection_with_ml(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return []

    frames = []
    labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        gray_frame = preprocess_frame(frame)

        # Example: Extract SIFT features
        keypoints, descriptors = extract_sift_features(gray_frame)
        if descriptors is not None:
            frames.append(descriptors.flatten())
            labels.append(1)  # 1 indicates shot boundary, you need to label your data accordingly

    cap.release()

    # Train SVM model
    svm_model = train_svm_model(frames, labels)

    # Classify shot boundaries
    shot_boundaries = classify_shot_boundaries(frames, labels, svm_model)

    return shot_boundaries


if __name__ == "__main__":
    # Replace with video path relative to you
    video_path = "C:\Users\paras\OneDrive\Documents\Frags AI Code\cod.mp4"  # Path to input video
    shot_boundaries = shot_boundary_detection(video_path)
    print(f"Shot boundary detection complete. {shot_boundaries}")
