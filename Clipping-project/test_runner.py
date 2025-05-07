import os
import logging
from preprocessing import extract_frames, process_frames, adjust_sample_interval, determine_chunk_size
from action_detection import extract_features
from audio_analysis import audio_analysis_pipeline
from clip_segmentation import segment_video_and_audio
from virality_ranking import get_video_clips_from_folder, rank_clips, copy_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = 'uploads'
VIDEO_FILE = 'test.mp4'
OUTPUT_FOLDER = 'output_segments'

if __name__ == "__main__":
    logging.info("Starting test runner for clipping tool...")

    # Step 1: Segment video into 1-minute clips (motion detected)
    input_video_path = os.path.join(UPLOAD_FOLDER, VIDEO_FILE)
    logging.info(f"Looking for video at: {input_video_path}")
    logging.info(f"Current working directory: {os.getcwd()}")

    if not os.path.exists(input_video_path):
        logging.error(f"Video file {input_video_path} not found. Make sure it's placed in the correct folder.")
        exit(1)

    logging.info("Segmenting the video into clips based on motion detection...")
    segment_video_and_audio(input_video_path, OUTPUT_FOLDER, segment_duration=60, max_segments=15)

    # Step 2: Get all segmented video clips
    video_clips = get_video_clips_from_folder(os.path.join(OUTPUT_FOLDER, "videos"))

    if not video_clips:
        logging.error("No video clips found after segmentation. Exiting.")
        exit(1)

    logging.info(f"Found {len(video_clips)} video clips for ranking.")

# Step 3: Rank clips by predicted virality
    logging.info("Ranking clips based on predicted virality...")
    # Define the YOLO model paths
    weights_path = r"D:\Fragss-AI-main\Fragss-AI-main\yolo.weights"  # Use raw string or double backslashes
    cfg_path = r"D:\Fragss-AI-main\Fragss-AI-main\yolov3.cfg"
    model_args = (weights_path, cfg_path)

    # Step 3: Rank clips by predicted virality
    logging.info("Ranking clips based on predicted virality...")
    ranked_clips = rank_clips(video_clips, model_args)

    logging.info("Ranking of video clips based on predicted virality:")
    for rank, clip in enumerate(ranked_clips, start=1):
        clip_path = clip[0]  # Assuming clip[0] is the path of the video clip
        logging.info(f"Rank {rank}: {clip_path}")  #

    # Step 4: Extract top 5 clip paths
    top_clip_paths = [clip[0] for clip in ranked_clips[:10]]

    # Step 5: Copy top clips to folder
    clip_virality_folder = os.path.join(OUTPUT_FOLDER, "videos", "clip_virality")
    copy_files(top_clip_paths, destination_folder="output_segments/videos/clip_virality")

    logging.info("\n✅ Done! Top 10 viral clips are ready inside 'output_segments/videos/clip_virality/'!")

