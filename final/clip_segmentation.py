# clip_segmentation.py

import cv2
import os
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Segment clips based on shot boundaries
def segment_clips(video_path, shot_boundaries, output_dir):
    """
    Segments video into clips based on detected shot boundaries.
    Args:
        video_path (str): Path to the original video file.
        shot_boundaries (list): List of frame indices where shots begin.
        output_dir (str): Directory to save segmented clips.
    """
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Starting video segmentation based on shot boundaries")
    for i, boundary in enumerate(shot_boundaries[:-1]):
        start_frame = boundary
        end_frame = shot_boundaries[i + 1]

        # Set the start position of the video capture to the start frame
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
        logging.info(f"Segment {i + 1} saved to {output_path}")

    cap.release()
    logging.info("Completed video segmentation")

# Extract audio segments based on shot boundaries
def extract_audio_segments(video_path, shot_boundaries, output_dir):
    """
    Extracts audio segments from the video based on shot boundaries.
    Args:
        video_path (str): Path to the original video file.
        shot_boundaries (list): List of frame indices where shots begin.
        output_dir (str): Directory to save audio segments.
    """
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
        logging.info(f"Audio segment {i + 1} saved to {output_path}")

# Combine segmented video and audio into final clips
def combine_video_audio(video_dir, audio_dir, output_dir):
    """
    Combines segmented video and audio files into final video clips.
    Args:
        video_dir (str): Directory containing video segments.
        audio_dir (str): Directory containing audio segments.
        output_dir (str): Directory to save final combined video-audio clips.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    for video_file, audio_file in zip(video_files, audio_files):
        video_path = os.path.join(video_dir, video_file)
        audio_path = os.path.join(audio_dir, audio_file)

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        final_clip = video_clip.set_audio(audio_clip)
        output_path = os.path.join(output_dir, video_file)

        final_clip.write_videofile(output_path, codec='libx264')
        logging.info(f"Combined video-audio clip saved to {output_path}")
