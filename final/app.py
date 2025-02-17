from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import logging
from final.preprocessing import preprocess_video_pipeline
from final.action_detection import extract_features
from final.audio_analysis import audio_analysis_pipeline
from final.shot_detection import detect_shot_boundaries
from clip_segmentation import segment_clips, extract_audio_segments, combine_video_audio
from final.virality_ranking import rank_clips, load_action_model
from final.subtitles import apply_subtitles_to_clips
from final.editing import edit_video
from final.background import generate_background
from final.voiceover import generate_voiceover
from final.aspect_ratio import enhance_video_aspect_ratio
from final.transcription import transcribe_video
from final.script import generate_stream_script

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        
        # User Script Generator
        script_generation = request.form.get("script_generation")
        if script_generation:
            return redirect(url_for('process_video', script_generation=script_generation))
        
        # Save user settings
        user_options = {
            "font": request.form.get("font"),
            "color": request.form.get("color"),
            "transition": request.form.get("transition"),
            "background_prompt": request.form.get("background_prompt"),
            "voiceover_text": request.form.get("voiceover_text"),
            "aspect_ratio": "aspect_ratio" in request.form,
            "transcription": "transcription" in request.form,
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
    
    script_generation = request.args.get("script_generation")
    
    # Use Script Generator
    if script_generation:
        script = generate_stream_script(script_generation)
        return render_template(
            "process.html",
            script=script
        )
    
    # Retrieve processing options
    processed_output = request.args.get('processed_output')
    font = request.args.get("font")
    color = request.args.get("color")
    transition = request.args.get("transition")
    background_prompt = request.args.get("background_prompt")
    voiceover_text = request.args.get("voiceover_text")
    aspect_ratio = request.args.get("aspect_ratio") == "True"
    transcription = request.args.get("transcription") == "True"
    
    if processed_output:
        logging.info("Step 2: Action Detection")
        actions_detected = extract_features(processed_output)
        
        logging.info("Step 3: Audio Analysis")
        audio_analysis_results = audio_analysis_pipeline(processed_output)
        
        logging.info("Step 4: Shot Detection")
        shot_boundaries = detect_shot_boundaries(processed_output, method='sift')
        
        logging.info("Step 5: Clip Segmentation")
        segment_clips(processed_output, shot_boundaries, OUTPUT_VIDEO_DIR)
        extract_audio_segments(processed_output, shot_boundaries, OUTPUT_AUDIO_DIR)
        combine_video_audio(OUTPUT_VIDEO_DIR, OUTPUT_AUDIO_DIR, FINAL_OUTPUT_DIR)
        
        # Step 6: Apply Selected Features
        clip_paths = [os.path.join(FINAL_OUTPUT_DIR, f) for f in sorted(os.listdir(FINAL_OUTPUT_DIR)) if f.endswith('.mp4')]

        if transcription:
            logging.info("Transcribing video...")
            transcriptions = [transcribe_video(clip, output_dir="transcriptions") for clip in clip_paths]
        
        if font and color:
            logging.info("Applying subtitles...")
            apply_subtitles_to_clips(clip_paths, font, color)
        
        if transition:
            logging.info("Applying transitions...")
            for clip in clip_paths:
                edit_video(clip_url=clip, title="Custom Edited Clip")
        
        if background_prompt:
            logging.info("Generating background...")
            generate_background(prompt=background_prompt)
        
        if voiceover_text:
            logging.info("Generating voiceover...")
            for clip in clip_paths:
                generate_voiceover(text=voiceover_text, output_path=f"{clip}_voiceover.mp3")
        
        if aspect_ratio:
            logging.info("Adjusting aspect ratio...")
            enhance_video_aspect_ratio(clip_paths)
        
        # Step 7: Virality Ranking
        logging.info("Ranking clips based on virality...")
        virality_model = joblib.load('virality_model.pkl')
        action_model = load_action_model('action_model.h5')
        ranked_clips = rank_clips(clip_paths, transcriptions if transcription else [], virality_model, action_model)
        
        # Render results
        return render_template(
            "process.html",
            script=script,
            actions=actions_detected, 
            audio=audio_analysis_results, 
            shots=shot_boundaries, 
            ranked_clips=ranked_clips
        )
    else:
        return "Error: Processed output not found"

if __name__ == "__main__":
    app.run(debug=True)
