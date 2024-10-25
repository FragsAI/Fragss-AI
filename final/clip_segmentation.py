import cv2
import os
import numpy as np
import moviepy.editor as mp
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from moviepy.editor import VideoFileClip
import json
from tkinter import filedialog, messagebox
import tkinter as tk
import logging
import pickle
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Segmentation Tool")

        self.video_path = None
        self.output_dir = None

        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.select_video_button = tk.Button(self.frame, text="Select Video", command=self.select_video)
        self.select_video_button.grid(row=0, column=0, padx=5, pady=5)

        self.select_output_button = tk.Button(self.frame, text="Select Output", command=self.select_output)
        self.select_output_button.grid(row=0, column=1, padx=5, pady=5)

        self.process_button = tk.Button(self.frame, text="Segment Video", command=self.segment_video, state=tk.DISABLED)
        self.process_button.grid(row=0, column=2, padx=5, pady=5)

        self.object_detection_var = tk.BooleanVar()
        self.object_detection_check = tk.Checkbutton(self.frame, text="Object Detection", variable=self.object_detection_var)
        self.object_detection_check.grid(row=1, column=0, padx=5, pady=5)

        self.audio_event_var = tk.BooleanVar()
        self.audio_event_check = tk.Checkbutton(self.frame, text="Audio Event Detection", variable=self.audio_event_var)
        self.audio_event_check.grid(row=1, column=1, padx=5, pady=5)

    def select_video(self):
        try:
            self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
            if self.video_path:
                self.process_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while selecting the video: {str(e)}")

    def select_output(self):
        try:
            self.output_dir = filedialog.askdirectory()
            if self.output_dir:
                self.process_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while selecting the output directory: {str(e)}")

    def segment_video(self):
        if self.video_path and self.output_dir:
            try:
                self.detect_shot_boundaries()
                self.generate_clips()
                if self.audio_event_var.get():
                    self.clip_audio()
                self.combine_clips()
                messagebox.showinfo("Success", "Video segmentation completed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showerror("Error", "Please select both a video file and an output directory.")

    def detect_shot_boundaries(self):
        self.shot_boundaries = detect_shot_boundaries(self.video_path, method='sift')

    def generate_clips(self):
        segment_clips(self.video_path, self.shot_boundaries, self.output_dir)

    def clip_audio(self):
        clip_audio(self.video_path, self.shot_boundaries, self.output_dir)

    def combine_clips(self):
        combine_video_audio(self.output_dir, self.output_dir, self.output_dir)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

# Function to load YOLO model for object detection
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# YOLO-based object detection
def yolo_object_detection(input_video_path, output_video_path):
    net, classes, output_layers = load_yolo_model()
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Multithreaded frame extraction
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

# Shot boundary detection based on histogram differences, SIFT, and frame differencing
def detect_shot_boundaries(video_path, method='sift', diff_threshold=50000, match_threshold=0.7, hist_diff_threshold=0.5, num_threads=4):
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

# Segment clips based on shot boundaries
def segment_clips(video_path, shot_boundaries, output_dir):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, boundary in enumerate(shot_boundaries[:-1]):
        start_frame = boundary
        end_frame = shot_boundaries[i + 1]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_path = os.path.join(output_dir, f"{video_filename}_segment_{i + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
    
    cap.release()
    logging.info(f"Saved segmented clips to {output_dir}")

# Extract audio segments based on shot boundaries
def extract_audio_segments(video_path, shot_boundaries, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    y, sr = librosa.load(video_path, sr=None, mono=True)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    for i, boundary in enumerate(shot_boundaries[:-1]):
        start_frame = boundary
        end_frame = shot_boundaries[i + 1]
        start_time = start_frame / sr
        end_time = end_frame / sr
        
        audio_segment = y[int(start_time * sr):int(end_time * sr)]
        output_path = os.path.join(output_dir, f"{video_filename}_audio_segment_{i + 1}.wav")
        librosa.output.write_wav(output_path, audio_segment, sr)
        logging.info(f"Saved audio segment {i + 1} to {output_path}")

# Combine segmented video and audio into final clips
def combine_video_audio(video_dir, audio_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    for video_file, audio_file in zip(video_files, audio_files):
        video_path = os.path.join(video_dir, video_file)
        audio_path = os.path.join(audio_dir, audio_file)
        
        video_clip = VideoFileClip(video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        
        final_clip = video_clip.set_audio(audio_clip)
        output_path = os.path.join(output_dir, video_file)
        
        final_clip.write_videofile(output_path, codec='libx264')
        logging.info(f"Saved combined video-audio file to {output_path}")

# Main function to launch the segmentation tool
def main():
    root = tk.Tk()
    app = VideoSegmentationTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
