import os
import cv2
import random
import numpy as np
import ffmpeg
import moviepy.editor as mp
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_motion(video_path, segment_duration=60, fps_threshold=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / fps_threshold)

    prev_frame = None
    motion_scores = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    total_segments = int(duration // segment_duration)

    logging.info(f"Analyzing motion across {total_segments} segments...")

    for segment_idx in tqdm(range(total_segments), desc="Detecting motion"):
        start_frame = int(segment_idx * segment_duration * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        motion_detected = 0
        for _ in range(int(segment_duration * fps / frame_skip)):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.sum(frame_diff) / frame_diff.size
                if motion_score > 10:
                    motion_detected += 1

            prev_frame = gray

        if motion_detected > 0:
            motion_scores[segment_idx] = motion_detected

    cap.release()
    return motion_scores

def segment_video_and_audio(video_path, output_dir, segment_duration=60, max_segments=30):
    if not os.path.exists(video_path):
        logging.error(f"Video file {video_path} not found.")
        return

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("Failed to open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        logging.error("Invalid video file or unreadable metadata.")
        cap.release()
        return

    duration = total_frames / fps
    total_segments = int(duration // segment_duration)

    if total_segments == 0:
        logging.warning("Video too short to segment. Skipping.")
        cap.release()
        return

    motion_scores = detect_motion(video_path, segment_duration)

    if not motion_scores:
        logging.warning("No motion detected in any segments. Exiting.")
        cap.release()
        return

    # Sort segments by motion detected (most to least motion)
    sorted_segments = sorted(motion_scores.items(), key=lambda x: x[1], reverse=True)
    selected_segments = [seg_idx for seg_idx, score in sorted_segments[:max_segments]]

    video_output_dir = os.path.join(output_dir, "videos")
    audio_output_dir = os.path.join(output_dir, "audios")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    logging.info(f"Processing {len(selected_segments)} action-rich segments.")

    segment_count = 0
    for seg_index in selected_segments:
        start_time = seg_index * segment_duration
        output_video_path = os.path.join(video_output_dir, f"{video_filename}_segment_{segment_count + 1}.mp4")
        output_audio_path = os.path.join(audio_output_dir, f"{video_filename}_segment_{segment_count + 1}.mp3")

        ffmpeg.input(video_path, ss=start_time, t=segment_duration).output(output_video_path, vcodec="libx264", acodec="aac").run(overwrite_output=True)

        video_clip = mp.VideoFileClip(output_video_path)
        video_clip.audio.write_audiofile(output_audio_path, codec="mp3")

        logging.info(f"Saved segment {segment_count + 1}: {output_video_path}")

        video_clip.close()
        segment_count += 1

    cap.release()
    logging.info(f"Completed video segmentation. Total segments created: {segment_count}")
