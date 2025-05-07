import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from clip_segmentation import segment_video_and_audio
from virality_ranking import rank_clips, copy_files  # Make sure to implement/import these

# === Config ===
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_segments'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
YOLO_WEIGHTS = r"D:\Fragss-AI-main\Fragss-AI-main\yolo.weights"
YOLO_CFG = r"D:\Fragss-AI-main\Fragss-AI-main\yolov3.cfg"

# === Ensure folders exist ===
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Set up Flask app ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Logger ===
logging.basicConfig(level=logging.INFO)

# === Helpers ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_segment', methods=['POST'])
def upload_and_segment_video():
    logging.info(f"request.files: {request.files}")
    logging.info(f"request.form: {request.form}")

    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        logging.info(f"Video uploaded to: {video_path}")

        try:
            # === Step 1: Segment video ===
            segment_video_and_audio(video_path, OUTPUT_FOLDER, segment_duration=60, max_segments=2)
            segmented_folder = os.path.join(OUTPUT_FOLDER, "videos")

            # === Step 2: Rank clips ===
            logging.info("Ranking clips based on predicted virality...")
            video_clips = [
                os.path.join(segmented_folder, f)
                for f in os.listdir(segmented_folder)
                if allowed_file(f)
            ]
            model_args = (YOLO_WEIGHTS, YOLO_CFG)
            ranked_clips = rank_clips(video_clips, model_args)

            # === Step 3: Copy top 10 clips ===
            top_clip_paths = [clip[0] for clip in ranked_clips[:10]]
            virality_folder = os.path.join(segmented_folder, "clip_virality")
            os.makedirs(virality_folder, exist_ok=True)
            copy_files(top_clip_paths, virality_folder)

            # === Step 4: Response ===
            return jsonify({
                'message': f'Video {filename} segmented and ranked successfully.',
                'output_folder': segmented_folder,
                'top_viral_clips': [
                    {'rank': i+1, 'clip_path': path} for i, path in enumerate(top_clip_paths)
                ]
            }), 200

        except Exception as e:
            logging.exception("Processing failed.")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Unsupported file type. Use .mp4, .avi, or .mov'}), 400

# === Run Flask app ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
