from flask import Flask, request, jsonify, send_from_directory
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

# Define upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check for valid file type
    if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        try:
            # Save the uploaded video file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logging.info(f"Video successfully saved at: {file_path}")
            
            # Return success response with the filename
            return jsonify({'status': 'success', 'filename': filename}), 200
        except Exception as e:
            logging.error(f"Error saving video file: {e}")
            return jsonify({'error': 'Failed to save video'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

    
# Endpoint to process the video asynchronously
@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    data = request.get_json()
    video_filename = data.get('video_filename')

    if not video_filename:
        return jsonify({'error': 'No video filename provided'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    logging.info(f"Processing video from path: {video_path}")

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found.'}), 404

    try:
        results = process_video(video_path)
        if not results:
            return jsonify({'status': 'success', 'results': []}), 200

        return jsonify({
            'status': 'success',
            'results': [{
                'clip': result['clip'],
                'virality_score': result['virality_score'],
                'final_video': result['final_video']
            } for result in results]
        }), 200
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return jsonify({'error': 'Failed to process video'}), 500


    
# Helper function to process video and return results
def process_video(video_path,model_size, device):
    audio, sr = extract_audio(video_path)
    loudest_times = audio_detection(audio, sr)
    clips = segment_video(video_path, loudest_times)
    save_clips(clips, OUTPUT_FOLDER)
    video_scores = process_videos_in_folder(UPLOAD_FOLDER)
    
    results = []
    for clip_path, score in video_scores:
        clip_file_path = os.path.join(UPLOAD_FOLDER, clip_path)
        extracted_audio = extract_audio_ffmpeg(clip_file_path)
        if extracted_audio:
            language, segments = transcribe_audio(extracted_audio, model_size, device)
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
    
# Endpoint to get virality scores for processed clips
@app.route('/get_scores', methods=['GET'])
def get_scores():
    try:
        video_scores = process_videos_in_folder(OUTPUT_FOLDER)
        return jsonify({'scores': video_scores}), 200
    except Exception as e:
        logging.error(f"Error fetching scores: {e}")
        return jsonify({'error': 'Failed to fetch scores'}), 500

# Endpoint to download processed clips
@app.route('/download_clip/<clip_filename>', methods=['GET'])
def download_clip(clip_filename):
    clip_path = os.path.join(app.config['OUTPUT_FOLDER'], clip_filename)
    
    if not os.path.exists(clip_path):
        return jsonify({'error': 'Clip not found.'}), 404

    return send_from_directory(os.path.abspath(app.config['OUTPUT_FOLDER']), clip_filename)


if __name__ == '__main__':
    app.run(debug=True)
