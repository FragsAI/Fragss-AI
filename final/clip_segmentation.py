import os
import logging
import cv2
import random
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def segment_clips(video_path, output_dir, segment_duration=60, max_segments=30):
    """
    Segments video into 1-minute clips, limiting to a maximum number of segments.
    """
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration=total_frames/fps
    total_segments=int(duration//segment_duration)
    possible_segments=list(range(total_segments))

    random.shuffle(possible_segments)
    selected_segments=possible_segments[:max_segments]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Starting video segmentation")
    segment_count = 0
    for seg_index in selected_segments:
        start_frame=int(seg_index*segment_duration*fps)
        end_frame = min(start_frame + int(fps * segment_duration),total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        output_path = os.path.join(output_dir, f"{video_filename}_segment_{segment_count + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        segment_count += 1
        logging.info(f"Segment {segment_count} saved to {output_path}")

    cap.release()
    logging.info(f"Completed video segmentation. Total segments created: {segment_count}")


def extract_audio_segments(video_path, output_dir, segment_duration=60, max_segments=30):
    """
    Extracts 1-minute audio segments from the video, limiting to 30 segments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_clip = VideoFileClip(video_path)
    audio = video_clip.audio
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    total_duration = video_clip.duration

    segment_count = 0
    for start_time in range(0, int(total_duration), segment_duration):
        if segment_count >= max_segments:
            break  # Stop after 30 segments
        
        end_time = min(start_time + segment_duration, total_duration)
        audio_segment = audio.subclip(start_time, end_time)
        output_path = os.path.join(output_dir, f"{video_filename}_audio_segment_{segment_count + 1}.wav")
        audio_segment.write_audiofile(output_path)
        
        segment_count += 1
        logging.info(f"Audio segment {segment_count} saved to {output_path}")

def process_video_for_action_detection(video_path, output_dir):
    """
    Processes video for action detection by segmenting video and extracting audio.
    """
    video_segments_dir = os.path.join(output_dir, "video_segments")
    audio_segments_dir = os.path.join(output_dir, "audio_segments")

    # Segment video and extract audio (limit to 30 segments)
    segment_clips(video_path, video_segments_dir)
    extract_audio_segments(video_path, audio_segments_dir)

    logging.info("Video processing for action detection completed.")


def combine_video_audio(video_dir, audio_dir, output_dir):
    """
    Combines segmented video and audio files into final 1-minute clips.
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

