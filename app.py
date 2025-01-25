from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import logging
import requests
from werkzeug.utils import secure_filename
from final_pipeline import (
    extract_frames, predict_actions, assess_video_quality, calculate_virality, normalize_scores,
    extract_audio, audio_detection, segment_video, save_clips, process_videos_in_folder,
    extract_audio_ffmpeg, transcribe_audio, generate_subtitle_file, add_subtitle_to_video,
    enhance_video_with_aspect_ratio
)
from final.voiceover import generate_voiceover

app = Flask(__name__,template_folder='final/templates')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define upload, output and voiceover folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
VOICEOVER_FOLDER = 'voiceovers'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VOICEOVER_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
FONT_COLORS = ['yellow', 'white', 'red', 'green', 'blue', 'pink', 'purple', 'orange', 'black', 'gray']
FONT_SIZES = [18, 24, 30, 36, 42, 48, 54, 60, 72, 84]
FONT_TYPES = {
    'Georgia' : 'C:/Windows/Fonts/georgiai.ttf',
    'Impact' : 'C:/Windows/Fonts/impact.ttf',
    'Candara': 'C:/Windows/Fonts/Candarai.ttf',
    'Perpetua': 'C:/Windows/Fonts/PERB____.TTF',
    'Rockwell' : 'C:/Windows/Fonts/ROCKI.TTF'
}

# Endpoint to upload any video 
@app.route('/', methods=['POST','GET'])
def upload_video():
     if request.method=='POST':
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
        return render_template('upload.html')
 
# Endpoint to process any video 
@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    data = request.get_json()
    
    video_filename = data.get('video_filename')
    font_color = data.get('font_color')  # No default value
    font_size = data.get('font_size')  # No default value
    font_type = data.get('font_type')  # No default value

    # Validate that all font choices are provided
    if not font_color or font_color not in FONT_COLORS:
        return jsonify({'error': 'Invalid or missing font color. Please choose from the following: ' + ', '.join(FONT_COLORS)}), 400
    if not font_size or font_size not in FONT_SIZES:
        return jsonify({'error': 'Invalid or missing font size. Please choose from the following: ' + ', '.join(map(str, FONT_SIZES))}), 400
    if not font_type or font_type not in FONT_TYPES:
        return jsonify({'error': 'Invalid or missing font type. Please choose from the following: ' + ', '.join(FONT_TYPES)}), 400

    if not video_filename:
        return jsonify({'error': 'No video filename provided'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    logging.info(f"Processing video from path: {video_path}")

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found.'}), 404

    try:
        results = process_video(video_path, font_color, font_size, font_type)
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
def process_video(video_path, font_color, font_size, font_type):
    audio, sr = extract_audio(video_path)
    loudest_times = audio_detection(audio, sr)
    clips = segment_video(video_path, loudest_times)
    save_clips(clips, OUTPUT_FOLDER)
    video_scores = process_videos_in_folder(OUTPUT_FOLDER)
    
    results = []
    for clip_path, score in video_scores:
        clip_file_path = os.path.join(OUTPUT_FOLDER, clip_path)

        # Extract audio from the current clip
        extracted_audio = extract_audio_ffmpeg(clip_file_path)
        if extracted_audio:

            # Transcribe the extracted audio to get language and segments
            language, segments = transcribe_audio(extracted_audio)
            if language and segments:

                # Enhance the video while maintaining aspect ratio
                enhanced_video = enhance_video_with_aspect_ratio(clip_file_path, OUTPUT_FOLDER)
                if enhanced_video:

                    # Generate subtitle file for the enhanced video
                    subtitle_file = generate_subtitle_file(enhanced_video, language, segments, font_color, font_size, font_type)
                    if subtitle_file:

                        # Add subtitles to the enhanced video and save the final video
                        final_video = add_subtitle_to_video(enhanced_video, subtitle_file, extracted_audio, font_color, font_size, font_type)
                        results.append({
                            'clip': clip_path,
                            'virality_score': score,
                            'final_video': final_video
                        })
    return results

# Endpoint to generate voiceovers for a text
@app.route('/generate_voiceover', methods=['POST'])
def generate_voiceover_endpoint():
    data = request.get_json()
    text = data.get('text')
    voice = data.get('voice', 'Jessica')  

    if not text:
        return jsonify({'error': 'No text provided for voiceover.'}), 400

    # Generate voiceover using the imported function
    output_filename = f"{voice}_voiceover.mp3"
    output_path = os.path.join(VOICEOVER_FOLDER, output_filename)

    result = generate_voiceover(text, voice, output_path)  # Call the imported function

    if result.startswith("Voiceover generated"):
        return jsonify({'status': 'success', 'file_path': output_filename}), 200
    else:
        return jsonify({'error': result}), 500
    
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

# Endpoint to download voiceovered clips
@app.route('/download_voiceover/<filename>', methods=['GET'])
def download_voiceover(filename):
    """
    Endpoint to download a generated voiceover file.
    """
    file_path = os.path.join(VOICEOVER_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Voiceover file not found.'}), 404

    return send_from_directory(os.path.abspath(VOICEOVER_FOLDER), filename)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
