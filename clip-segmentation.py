import cv2
import numpy as np
import os
import logging
import pickle
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames_multithreaded(video_path, num_threads=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def process_frame(frame_index):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame_index, gray
        return frame_index, None
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_frame, i) for i in range(total_frames)]
        frames = {i: f.result()[1] for i, f in enumerate(futures) if f.result()[1] is not None}
    
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

    Args:
        video_path (str): Path to the video file.
        method (str): Method for shot detection ('sift', 'diff', 'hist').
        diff_threshold (int): Threshold for frame differencing method.
        match_threshold (float): Threshold for SIFT method.
        hist_diff_threshold (float): Threshold for histogram difference method.
        num_threads (int): Number of threads to use for frame extraction.
    
    Returns:
        list: List of frame indices where shot boundaries were detected.
    """
    frames = extract_frames_multithreaded(video_path, num_threads=num_threads)
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

def segment_clips(video_path, shot_boundaries, output_folder):
    """
    Segments a video into clips based on detected shot boundaries.

    Args:
        video_path (str): Path to the video file.
        shot_boundaries (list): List of frame indices where shot boundaries were detected.
        output_folder (str): Directory to save the segmented clips.
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

def optimize_model(X, y):
    """
    Optimizes a model using GridSearchCV.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
    
    Returns:
        estimator: Best estimator found by GridSearchCV.
    """
    pipeline = make_pipeline(StandardScaler(), SVR())
    param_grid = {
        'svr__C': [0.1, 1, 10],
        'svr__gamma': ['scale', 'auto'],
        'svr__kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def load_data(file_path):
    """
    Loads data from a file.

    Args:
        file_path (str): Path to the data file.
    
    Returns:
        tuple: Feature matrix and target vector.
    """
    return np.load(file_path)

def save_model(model, file_path):
    """
    Saves a model to a file.

    Args:
        model (estimator): Model to save.
        file_path (str): Path to save the model file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    video_path = "input_video.mp4"
    output_folder = "clips"
    
    logging.info("Starting shot boundary detection")
    shot_boundaries = detect_shot_boundaries(video_path, method='sift')
    
    logging.info("Starting clip segmentation")
    segment_clips(video_path, shot_boundaries, output_folder)
    
    logging.info("Starting model optimization")
    data_file_path = "training_data.npy"
    model_file_path = "optimized_model.pkl"
    
    X, y = load_data(data_file_path)
    best_model = optimize_model(X, y)
    save_model(best_model, model_file_path)
    logging.info("Model optimized and saved")

if __name__ == "__main__":
    main()
