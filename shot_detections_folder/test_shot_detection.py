import cv2
import os
from shot_boundary_detection import detect_shot_boundaries
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

# Add a thread lock for video capture operations
video_lock = threading.Lock()

def test_shot_detection(video_path, output_folder="shot_detection_results"):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get shot boundaries
    print("Detecting shot boundaries...")
    shot_boundaries = detect_shot_boundaries(video_path)
    print(f"Found {len(shot_boundaries)} shot boundaries")
    
    # Open video with thread safety
    with video_lock:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
    def process_boundary(boundary_info):
        i, timestamp_ms = boundary_info
        frame_number = int((timestamp_ms / 1000.0) * fps)
        
        for offset in [-5, 0, 5]:
            target_frame = max(0, frame_number + offset)
            
            # Thread-safe video frame capture
            with video_lock:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                
                if ret:
                    text = f"Frame: {target_frame}, Time: {timestamp_ms/1000:.2f}s"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    output_path = os.path.join(output_folder, f"boundary_{i+1}_offset_{offset}.jpg")
                    cv2.imwrite(output_path, frame)
    
    # Process boundaries using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_boundary, enumerate(shot_boundaries))
    
    # Safely release video capture
    with video_lock:
        cap.release()
    
    # Create a summary file
    with open(os.path.join(output_folder, "shot_boundaries_summary.txt"), "w") as f:
        f.write("Shot Boundaries Summary\n")
        f.write("=====================\n\n")
        f.write(f"Total shots detected: {len(shot_boundaries)}\n\n")
        f.write("Timestamps (seconds):\n")
        for i, ms in enumerate(shot_boundaries):
            f.write(f"Shot {i+1}: {ms/1000:.2f}s\n")

def create_preview_video(video_path, shot_boundaries, output_path="shot_detection_preview.mp4"):
    with video_lock:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer with MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        boundary_frames = [int((ts / 1000.0) * fps) for ts in shot_boundaries]
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in boundary_frames:
                frame = cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
                cv2.putText(frame, "SHOT BOUNDARY", (width//4, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()

if __name__ == "__main__":
    video_path = "/Users/rnzgrd/Downloads/SOLO_VS_SQUAD_34_KILLS_FULL_GAMEPLAY_CALL_OF_DUTY_MOBILE_BATTLE_ROYALE.mp4"
    
    # Run tests and create visualizations
    test_shot_detection(video_path)
    
    # Get boundaries again for preview video
    boundaries = detect_shot_boundaries(video_path)
    create_preview_video(video_path, boundaries)
    
    print("\nTesting complete! Check the shot_detection_results folder for:")
    print("1. Frame captures around each detected boundary")
    print("2. Summary text file with timestamps")
    print("3. Preview video with boundary markers") 