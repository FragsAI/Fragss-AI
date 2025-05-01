import os
import logging
from preprocessing_final import extract_frames, process_frames, adjust_sample_interval, determine_chunk_size
from action_detection import extract_features
from audio_analysis import audio_analysis_pipeline
from clip_segmentation import segment_video_and_audio
from virality_ranking import get_video_clips_from_folder, rank_clips, copy_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
UPLOAD_FOLDER = 'uploads'
VIDEO_FILE = 'my_stream_video.mp4'  # change this to your uploaded file name
OUTPUT_FOLDER = 'output_segments'

if __name__ == "__main__":
    logging.info("Starting test runner for clipping tool...")

    # Step 1: Segment video into 1-minute clips (motion detected)
    input_video_path = os.path.join(UPLOAD_FOLDER, VIDEO_FILE)
    logging.info("Segmenting the video into clips based on motion detection...")
    segment_video_and_audio(input_video_path, OUTPUT_FOLDER, segment_duration=60, max_segments=30)

    # Step 2: Get all segmented video clips
    video_clips = get_video_clips_from_folder(os.path.join(OUTPUT_FOLDER, "videos"))

    if not video_clips:
        logging.error("No video clips found after segmentation. Exiting.")
        exit(1)

    logging.info(f"Found {len(video_clips)} video clips for ranking.")

    # Step 3: Rank clips by predicted virality (using YOLO detections)
    logging.info("Ranking clips based on predicted virality...")
    ranked_clips = rank_clips(video_clips, None)

    # Step 4: Copy Top 20 Clips to 'clip_virality' folder
    top_clips = ranked_clips[:20]
    copy_files(top_clips)

    logging.info("\n✅ Done! Top 20 viral clips are ready inside 'output_segments/videos/clip_virality/'!")