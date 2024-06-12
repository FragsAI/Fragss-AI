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
from moviepy.editor import VideoFileClip, concatenate_videoclips
import json
from tkinter import filedialog, messagebox
import tkinter as tk
from functools import partial

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
        """
        Detect shot boundaries in the input video using the shot_boundary_detection function.
        """
        self.shot_boundaries = shot_boundary_detection(self.video_path)

    def generate_clips(self):
        """
        Generate video clips based on the detected shot boundaries.
        """
        clip_video(self.video_path, self.shot_boundaries, self.output_dir)

    def clip_audio(self):
        """
        Generate audio clips based on the detected shot boundaries.
        """
        clip_audio(self.video_path, self.shot_boundaries, self.output_dir)

    def combine_clips(self):
        """
        Combine the video and audio clips into the final output files.
        """
        combine_video_audio(self.output_dir, self.output_dir, self.output_dir)

    def on_closing(self):
        """
        Handle the closing of the application window and prompt the user for confirmation.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()


def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

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


def shot_boundary_detection(video_path, threshold=100):
    """
    Detect shot boundaries in a video based on histogram differences between frames.

    Args:
    - video_path (str): Path to the input video file.
    - threshold (int): Threshold for detecting shot boundaries.

    Returns:
    - shot_boundaries (list): List of timestamps representing shot boundaries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return []
    
    # Initialize variables
    prev_frame_hist = None
    shot_boundaries = [0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Check if it's the first frame
        if prev_frame_hist is None:
            prev_frame_hist = hist
            continue
        
        # Calculate histogram difference
        hist_diff = cv2.compareHist(prev_frame_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        
        # If the difference is above the threshold, it indicates a shot boundary
        if hist_diff > threshold:
            shot_boundaries.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Update previous frame histogram
        prev_frame_hist = hist
    
    cap.release()
    
    return shot_boundaries

def clip_video(video_path, shot_boundaries, output_dir):
    """
    Clip the input video based on detected shot boundaries and save each segment as a separate file.

    Args:
    - video_path (str): Path to the input video file.
    - shot_boundaries (list): List of timestamps representing shot boundaries.
    - output_dir (str): Directory to save the clipped video segments.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Iterate over shot boundaries and create clips
    for i, boundary in enumerate(shot_boundaries):
        start_time = boundary / 1000  # convert milliseconds to seconds
        end_time = (shot_boundaries[i + 1] / 1000)-0.1 if i < len(shot_boundaries) - 1 else None
        
        # Set capture to start at the beginning of the clip
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        # Initialize VideoWriter for the clip
        clip_name = os.path.join(output_dir, f"clip_{i}.mp4")
        out = cv2.VideoWriter(clip_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Read frames and write to the clip until the end time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write frame to the clip
            out.write(frame)
            
            # Check if reached the end time of the clip
            if end_time and cap.get(cv2.CAP_PROP_POS_MSEC) >= end_time * 1000:
                break
        
        # Release VideoWriter and move to the next clip
        out.release()

    cap.release()

def clip_audio(video_path, shot_boundaries, output_dir):
    """
    Clip the audio of the input video based on detected shot boundaries and save each segment as a separate file.

    Args:
    - video_path (str): Path to the input video file.
    - shot_boundaries (list): List of timestamps representing shot boundaries.
    - output_dir (str): Directory to save the clipped audio segments.
    """
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    
    # Iterate over shot boundaries and create audio clips
    for i, boundary in enumerate(shot_boundaries):
        start_time = boundary / 1000  # convert milliseconds to seconds
        end_time = (shot_boundaries[i + 1] / 1000)-0.1 if i < len(shot_boundaries) - 1 else None
        
        # Clip the audio
        audio_clip = audio.subclip(start_time, end_time)
        
        # Write audio clip to file
        audio_clip.write_audiofile(os.path.join(output_dir, f"audio_clip_{i}.mp3"))

def combine_video_audio(video_dir, audio_dir, output_dir):
    """
    Combine the clipped video and audio segments into final clips.

    Args:
    - video_dir (str): Directory containing the clipped video segments.
    - audio_dir (str): Directory containing the clipped audio segments.
    - output_dir (str): Directory to save the combined video clips.
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
    
    # Ensure equal number of video and audio clips
    num_clips = min(len(video_files), len(audio_files))
    
    for i in range(num_clips):
        video_path = os.path.join(video_dir, f"clip_{i}.mp4")
        audio_path = os.path.join(audio_dir, f"audio_clip_{i}.mp3")
        
        # Load video clip
        video_clip = mp.VideoFileClip(video_path)
        
        try:
            # Load audio clip
            audio_clip = mp.AudioFileClip(audio_path)
            
            # Set audio for video clip
            video_clip = video_clip.set_audio(audio_clip)
            
            # Write combined clip to file
            combined_clip_path = os.path.join(output_dir, f"combined_clip_{i}.mp4")
            video_clip.write_videofile(combined_clip_path, codec='libx264', audio_codec='aac')
            
            print(f"Combined clip {i} successfully created.")
        except Exception as e:
            print(f"Error combining audio and video for clip {i}: {e}")
        
        # Remove both video and audio clips
        os.remove(video_path)
        os.remove(audio_path)

def detect_audio_events(video_path, threshold=0.02):
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    
    y, sr = librosa.load(audio_path)
    audio_events = []

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    for frame in onset_frames:
        event_time = librosa.frames_to_time(frame, sr=sr)
        audio_events.append(event_time * 1000)  # Convert to milliseconds

    os.remove(audio_path)
    return audio_events

class VideoActionDataset(Dataset):
    def __init__(self, video_files, labels, transform=None):
        self.video_files = video_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video = mp.VideoFileClip(self.video_files[idx])
        frames = [frame for frame in video.iter_frames()]
        label = self.labels[idx]
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = np.stack(frames)
        frames = torch.tensor(frames).permute(3, 0, 1, 2)  # Move channels to front
        return frames, label

def train_action_recognition_model(data_dir):
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mp4')]
    labels_file = os.path.join(data_dir, 'labels.json')
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Transform and DataLoader
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    dataset = VideoActionDataset(video_files, labels, transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Model
    model = models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(set(labels.values())))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    
    torch.save(model.state_dict(), os.path.join(data_dir, 'action_model.pth'))

def infer_and_label(data_dir):
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mp4')]
    labels_file = os.path.join(data_dir, 'labels.json')
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    model = models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(set(labels.values())))
    model.load_state_dict(torch.load(os.path.join(data_dir, 'action_model.pth')))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    for video_file in video_files:
        video = mp.VideoFileClip(video_file)
        frames = [frame for frame in video.iter_frames()]
        frames = [transform(frame) for frame in frames]
        frames = np.stack(frames)
        frames = torch.tensor(frames).permute(3, 0, 1, 2)  # Move channels to front
        with torch.no_grad():
            output = model(frames.unsqueeze(0))
            _, predicted = torch.max(output, 1)
        
        action = labels[str(predicted.item())]
        labeled_clip = video.subclip(0, 10).set_duration(video.duration).set_text(action, fontsize=50, color='white', bg_color='black', text_position=('center', 'top'))
        labeled_clip.write_videofile(os.path.join(data_dir, f"labeled_{os.path.basename(video_file)}"), codec='libx264', audio_codec='aac')


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSegmentationTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
