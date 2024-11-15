# app.py

from flask import Flask, request, render_template, redirect, url_for
import os
from preprocessing import preprocess_video_pipeline
from action_detection import extract_features
from audio_analysis import audio_analysis_pipeline
from shot_detection import detect_shot_boundaries
from clip_segmentation import segment_clips, extract_audio_segments, combine_video_audio
from virality_ranking import rank_clips, load_action_model
import load

# Importing new modules
from subtitles import apply_subtitles
from editing import apply_transitions
from background import change_background
from voiceover import add_voiceover
from aspect_ratio import adjust_aspect_ratio
from transcription import generate_transcription
from script import generate_script

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_frames'
OUTPUT_VIDEO_DIR = 'segmented_videos'
OUTPUT_AUDIO_DIR = 'segmented_audio'
FINAL_OUTPUT_DIR = 'final_clips'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        # Save user settings
        user_options = {
            "font": request.form.get("font"),
            "color": request.form.get("color"),
            "transition": request.form.get("transition"),
            "background": "background" in request.form,
            "voiceover": "voiceover" in request.form,
            "aspect_ratio": "aspect_ratio" in request.form,
            "transcription": "transcription" in request.form,
            "script_generation": "script_generation" in request.form
        }

        file = request.files['file']
        if file:
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)
            processed_output = preprocess_video_pipeline(video_path, PROCESSED_FOLDER)
            return redirect(url_for('process_video', processed_output=processed_output, **user_options))
    return render_template("upload.html")

@app.route("/process", methods=["GET"])
def process_video():
    # Retrieve processing options
    processed_output = request.args.get('processed_output')
    font = request.args.get("font")
    color = request.args.get("color")
    transition = request.args.get("transition")
    background = request.args.get("background") == "True"
    voiceover = request.args.get("voiceover") == "True"
    aspect_ratio = request.args.get("aspect_ratio") == "True"
    transcription = request.args.get("transcription") == "True"
    script_generation = request.args.get("script_generation") == "True"
    
    if processed_output:
        # Step 2: Action Detection
        actions_detected = extract_features(processed_output)
        
        # Step 3: Audio Analysis
        audio_analysis_results = audio_analysis_pipeline(processed_output)
        
        # Step 4: Shot Detection
        shot_boundaries = detect_shot_boundaries(processed_output, method='sift')
        
        # Step 5: Clip Segmentation
        segment_clips(processed_output, shot_boundaries, OUTPUT_VIDEO_DIR)
        extract_audio_segments(processed_output, shot_boundaries, OUTPUT_AUDIO_DIR)
        combine_video_audio(OUTPUT_VIDEO_DIR, OUTPUT_AUDIO_DIR, FINAL_OUTPUT_DIR)
        
        # Step 6: Apply Selected Features
        clip_paths = [os.path.join(FINAL_OUTPUT_DIR, f) for f in sorted(os.listdir(FINAL_OUTPUT_DIR)) if f.endswith('.mp4')]
        
        if transcription:
            transcriptions = generate_transcription(processed_output)
        
        if script_generation:
            script = generate_script(processed_output)
        
        if font and color:
            apply_subtitles(clip_paths, font, color)
        
        if transition:
            apply_transitions(clip_paths, transition)
        
        if background:
            change_background(clip_paths)
        
        if voiceover:
            add_voiceover(clip_paths)
        
        if aspect_ratio:
            adjust_aspect_ratio(clip_paths)
        
        # Step 7: Virality Ranking
        virality_model = load('virality_model.pkl')
        action_model = load_action_model('action_model.h5')
        ranked_clips = rank_clips(clip_paths, transcriptions if transcription else [], virality_model, action_model)
        
        # Render results
        return render_template("process.html", actions=actions_detected, audio=audio_analysis_results, shots=shot_boundaries, ranked_clips=ranked_clips)
    else:
        return "Error: Processed output not found"

if __name__ == "__main__":
    app.run(debug=True)
