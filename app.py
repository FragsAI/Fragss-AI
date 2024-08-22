from flask import Flask, request, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from final_pipeline import (
    extract_frames, predict_actions, assess_video_quality, calculate_virality, normalize_scores,
    extract_audio, audio_detection, segment_video, save_clips, process_videos_in_folder,
    extract_audio_ffmpeg, transcribe_audio, generate_subtitle_file, add_subtitle_to_video,
    enhance_video_with_aspect_ratio
)

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to process video and return results
def process_video(video_path):
    audio, sr = extract_audio(video_path)
    loudest_times = audio_detection(audio, sr)
    clips = segment_video(video_path, loudest_times)
    save_clips(clips, UPLOAD_FOLDER)
    video_scores = process_videos_in_folder(UPLOAD_FOLDER)
    
    results = []
    for clip_path, score in video_scores:
        clip_file_path = os.path.join(UPLOAD_FOLDER, clip_path)
        extracted_audio = extract_audio_ffmpeg(clip_file_path)
        if extracted_audio:
            language, segments = transcribe_audio(extracted_audio)
            if language and segments:
                enhanced_video = enhance_video_with_aspect_ratio(clip_file_path, clip_file_path)
                if enhanced_video:
                    subtitle_file = generate_subtitle_file(enhanced_video, language, segments)
                    if subtitle_file:
                        final_video = add_subtitle_to_video(enhanced_video, subtitle_file, extracted_audio)
                        results.append({
                            'clip': clip_path,
                            'virality_score': score,
                            'final_video': final_video
                        })
    return results

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"Video saved at: {file_path}")

        try:
            results = process_video(file_path)
            return jsonify({'status': 'success', 'results': results}), 200
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            return jsonify({'error': 'Failed to process video'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
