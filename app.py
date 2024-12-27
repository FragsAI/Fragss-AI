from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
import datetime
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from celery import Celery
import uuid
from typing import Dict, List, Optional
import warnings

# Import local modules
from final_pipeline import (
    extract_frames, predict_actions, assess_video_quality, calculate_virality, normalize_scores,
    extract_audio, audio_detection, segment_video, save_clips, process_videos_in_folder,
    extract_audio_ffmpeg, transcribe_audio, generate_subtitle_file, add_subtitle_to_video,
    enhance_video_with_aspect_ratio
)

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# App configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Store task status
task_status: Dict[str, Dict] = {}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@celery.task(bind=True)
def process_video_task(self, video_path: str, task_id: str) -> Dict:
    try:
        task_status[task_id]['status'] = 'PROCESSING'
        
        # Extract audio and detect segments
        self.update_state(state='PROCESSING', meta={'step': 'Extracting audio'})
        audio, sr = extract_audio(video_path)
        
        self.update_state(state='PROCESSING', meta={'step': 'Detecting segments'})
        loudest_times = audio_detection(audio, sr)
        
        # Process video segments
        self.update_state(state='PROCESSING', meta={'step': 'Processing segments'})
        clips = segment_video(video_path, loudest_times)
        clips_folder = os.path.join(PROCESSED_FOLDER, task_id)
        os.makedirs(clips_folder, exist_ok=True)
        save_clips(clips, clips_folder)
        
        # Process individual clips
        self.update_state(state='PROCESSING', meta={'step': 'Analyzing clips'})
        video_scores = process_videos_in_folder(clips_folder)
        
        results = []
        for clip_path, score in video_scores:
            clip_file_path = os.path.join(clips_folder, clip_path)
            
            # Extract and process audio
            extracted_audio = extract_audio_ffmpeg(clip_file_path)
            if extracted_audio:
                language, segments = transcribe_audio(extracted_audio)
                if language and segments:
                    # Enhance video and add subtitles
                    enhanced_video = enhance_video_with_aspect_ratio(clip_file_path, clip_file_path)
                    if enhanced_video:
                        subtitle_file = generate_subtitle_file(enhanced_video, language, segments)
                        if subtitle_file:
                            final_video = add_subtitle_to_video(enhanced_video, subtitle_file, extracted_audio)
                            results.append({
                                'clip_id': os.path.basename(clip_path),
                                'virality_score': float(score),
                                'final_video': final_video,
                                'language': language
                            })
        
        task_status[task_id]['status'] = 'COMPLETED'
        task_status[task_id]['results'] = results
        return {'status': 'success', 'results': results}
        
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        task_status[task_id]['status'] = 'FAILED'
        task_status[task_id]['error'] = str(e)
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        task_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)
        
        # Initialize task status
        task_status[task_id] = {
            'status': 'PENDING',
            'filename': filename,
            'created_at': str(datetime.datetime.now())
        }
        
        # Start async processing
        task = process_video_task.delay(file_path, task_id)
        
        return jsonify({
            'status': 'success',
            'message': 'Video upload successful',
            'task_id': task_id
        }), 202
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large'}), 413
    except Exception as e:
        logging.error(f"Error in upload: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task_info = task_status.get(task_id)
    if not task_info:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task_info), 200

@app.route('/clip/<task_id>/<clip_id>', methods=['GET'])
def get_clip(task_id, clip_id):
    try:
        clip_path = os.path.join(PROCESSED_FOLDER, task_id, clip_id)
        if not os.path.exists(clip_path):
            return jsonify({'error': 'Clip not found'}), 404
        
        return send_file(clip_path, mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Error serving clip: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
