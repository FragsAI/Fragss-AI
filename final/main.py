# main.py

from preprocessing import preprocess_video
from clip_segmentation import segment_clips
from shot_detection import detect_shots
from audio_analysis import analyze_audio
from action_detection import detect_actions
from virality_ranking import rank_clips_by_virality

def process_video(input_video_path):
    # Step 1: Preprocess the video
    preprocessed_video = preprocess_video(input_video_path)
    
    # Step 2: Segment the video into clips
    clips = segment_clips(preprocessed_video)
    
    # Step 3: Perform shot detection on each clip
    detected_shots = [detect_shots(clip) for clip in clips]
    
    # Step 4: Analyze audio in each clip
    audio_analysis_results = [analyze_audio(clip) for clip in detected_shots]
    
    # Step 5: Detect actions in each clip
    action_detection_results = [detect_actions(clip) for clip in audio_analysis_results]
    
    # Step 6: Rank clips by virality
    ranked_clips = rank_clips_by_virality(action_detection_results)
    
    # Step 7: Select the top 20 clips
    top_20_clips = ranked_clips[:20]
    
    return top_20_clips

if __name__ == "__main__":
    input_video_path = "path/to/your/video.mp4"
    top_clips = process_video(input_video_path)
    
    # Save or output the top clips
    for i, clip in enumerate(top_clips):
        clip.save(f"output_clip_{i+1}.mp4")
    print("Top 20 clips have been saved with virality ranking.")
