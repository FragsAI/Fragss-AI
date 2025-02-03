import os
import cv2
import random
import numpy as np
import ffmpeg
import moviepy.editor as mp
import logging

def detect_motion(video_path, segment_duration=60, fps_threshold=10):
    """
    Detects action & motion using frame differencing and returns timestamps of action-heavy segments.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / fps_threshold)  # Reduce processing load

    prev_frame = None
    motion_timestamps = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    total_segments = int(duration // segment_duration)

    logging.info(f"Analyzing motion in {total_segments} segments...")

    for segment_idx in range(total_segments):
        start_frame = int(segment_idx * segment_duration * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_detected = False
        for _ in range(int(segment_duration * fps / frame_skip)):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.sum(frame_diff) / frame_diff.size
                
                if motion_score > 10:  # Threshold to detect motion
                    motion_detected = True
                    break
            
            prev_frame = gray

        if motion_detected:
            motion_timestamps.append(segment_idx)

    cap.release()
    return motion_timestamps

def segment_video_and_audio(video_path, output_dir, segment_duration=60, max_segments=30):
    """
    Segments video into 1-minute clips, extracts audio, and filters using motion detection.
    Stores video and audio in separate folders.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        logging.warning("Video is too short to segment. Skipping.")
        cap.release()
        return

    # Detect motion and select relevant segments
    motion_segments = detect_motion(video_path, segment_duration)
    if not motion_segments:
        logging.warning("No motion detected in any segments. Exiting.")
        cap.release()
        return

    random.shuffle(motion_segments)
    selected_segments = motion_segments[:min(len(motion_segments), max_segments)]

    # Create separate folders for videos and audios
    video_output_dir = os.path.join(output_dir, "videos")
    audio_output_dir = os.path.join(output_dir, "audios")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    logging.info(f"Processing {len(selected_segments)} segments with detected motion.")

    segment_count = 0
    for seg_index in selected_segments:
        start_time = seg_index * segment_duration
        output_video_path = os.path.join(video_output_dir, f"{video_filename}_segment_{segment_count + 1}.mp4")
        output_audio_path = os.path.join(audio_output_dir, f"{video_filename}_segment_{segment_count + 1}.mp3")

        # Extract Video
        ffmpeg.input(video_path, ss=start_time, t=segment_duration).output(output_video_path, vcodec="libx264", acodec="aac").run(overwrite_output=True)

        # Extract Audio
        video_clip = mp.VideoFileClip(output_video_path)
        video_clip.audio.write_audiofile(output_audio_path, codec="mp3")

        logging.info(f"Saved segment {segment_count + 1}:")
        logging.info(f"  Video: {output_video_path}")
        logging.info(f"  Audio: {output_audio_path}")

        # Close the video clip after processing
        video_clip.close()
        segment_count += 1

    cap.release()
    logging.info(f"Completed video segmentation. Total action-based segments created: {segment_count}")


# Example Usage
# video_file = "3.mp4"
# output_directory = "output_segments"
# segment_video_and_audio(video_file, output_directory)
