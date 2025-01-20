import cv2
import numpy as np
import os
import logging
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.config import change_settings
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import datetime as dt
from sklearn.metrics import classification_report
from scipy.stats import variation
import ffmpeg
from faster_whisper import WhisperModel
import pysrt
import math

import warnings
warnings.filterwarnings('ignore',category=UserWarning, module="moviepy")

'''
`moviepy` version needs to be 1.0.3
If unable to import `ffmpeg` then install by running `pip install ffmpeg-python`
If unable to install `pysrt` then install by running `pip install pysrt`
If unable to install `WhisperModel` then istall `faster_whisper` by running `! pip install faster_whisper`; then import `faster_whisper` and `WhisperModel`

`numpy` vesrion needs to be 1.24, run `pip install numpy==1.24` to install and restart the kernel and import numpy
`keras` vesrion needs to be 3.5.0, run  `pip install keras==3.5.0` to install and restart the kernel and import keras
`tensorflow` vesrion needs to be 2.17.1, run `pip install tensorflow==2.17.1` to install and restart the kernel and import tensorflow
`h5py` vesrion needs to be 3.12.1, run `pip install h5py==3.12.1` to install and restart the kernel and import h5py (run `import h5py`)

'''
imagemagick_path=input('Input full path of "magick.exe": ')
change_settings({"IMAGEMAGICK_BINARY":imagemagick_path}) #Adjust as per your system's PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
TIMESTEPS = 10  # Number of frames to consider in each sequence
MAX_PIXEL_VALUE = 255
BATCH_SIZE = 100
NO_OF_CHANNELS = 3
MODEL_PATH = input('Input your model path: ')

# Function to extract frames from a video
def extract_frames(video_path):
    frames_list = []
    videoObj = cv2.VideoCapture(video_path)
    logging.info(f"Extracting frames from video: {video_path}")
    while True:
        success, image = videoObj.read()
        if not success:
            break
        resized_frame = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / MAX_PIXEL_VALUE
        frames_list.append(normalized_frame)
    videoObj.release()
    while len(frames_list) < TIMESTEPS:
        frames_list.append(frames_list[-1])
    return np.array(frames_list[:TIMESTEPS])

# Function to extract audio from video
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    audio, sr = librosa.load(audio_path, sr=None)
    os.remove(audio_path)
    return audio, sr

# Use audio to find times for segmentation
def audio_detection(audio, sr, num_clips=5, clip_length=30):
    frame_length = sr * clip_length
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length)[0]
    indices = np.argsort(rms)[-num_clips:]
    times = indices * clip_length
    return times

# Function to segment video
def segment_video(video_path, events, segment_duration=15):
    video = VideoFileClip(video_path)
    clips = []
    for event in events:
        start_time = max(0, event - segment_duration / 2)
        end_time = min(video.duration, event + segment_duration / 2)
        if end_time - start_time < segment_duration:
            start_time = max(0, end_time - segment_duration)
        clip = video.subclip(start_time, end_time)
        clips.append(clip)
    return clips

# Function to save clips
def save_clips(clips, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, clip in enumerate(clips):
        output_path = os.path.join(output_dir, f"clip_{i + 1}.mp4")
        clip.write_videofile(output_path)

# Load pre-trained action detection model
logging.info(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)

# Function to predict actions in frames
def predict_actions(frames):
    features = np.array(frames).reshape(1, TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS)
    logging.info(f"Predicting actions for frames with shape: {features.shape}")
    
    predictions = model.predict(features)
    return predictions

# Function to assess video quality
def assess_video_quality(frames):
    brightness_scores = []
    blur_scores = []
    
    for frame in frames:
        gray = cv2.cvtColor((frame * MAX_PIXEL_VALUE).astype('uint8'), cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_scores.append(brightness)
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(laplacian_var)
    
    mean_brightness = np.mean(brightness_scores)
    mean_blur = np.mean(blur_scores)
    
    return mean_brightness, mean_blur

# Function to calculate virality score
def calculate_virality(predictions, quality_metrics):
    mean_confidence = np.mean(np.max(predictions, axis=1))
    variance_confidence = variation(np.max(predictions, axis=1))
    low_confidence_penalty = np.sum(np.max(predictions, axis=1) < 0.5)
    
    mean_brightness, mean_blur = quality_metrics
    
    score = (mean_confidence * 100) - (variance_confidence * 10) - (low_confidence_penalty * 5)
    score += mean_brightness * 0.01
    score -= mean_blur * 0.01
    
    return score

# Function to normalize scores
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) * 99 + 1 for score in scores]
    return normalized_scores

# Function to add subtitles to video clips
def extract_audio_ffmpeg(input_video):
    try:
        input_video_name = os.path.splitext(os.path.basename(input_video))[0]
        extracted_audio = os.path.join(os.path.dirname(input_video), f"audio-{input_video_name}.wav")
        # Ensure the directory is properly handled
        if not os.path.dirname(extracted_audio):
            extracted_audio = os.path.join(os.getcwd(), f"audio-{input_video_name}.wav")
        stream = ffmpeg.input(input_video)
        stream = ffmpeg.output(stream, extracted_audio)
        ffmpeg.run(stream, overwrite_output=True)
        logging.info(f"Audio extracted: {extracted_audio}")
        return extracted_audio
        '''
        If any error such as 'WindowsError: [Error 2] The system cannot find the file specified' encounters then run below line of codes: 
        pip uninstall ffmpeg-python
        conda install ffmpeg        # only if you use jupyter notebook
        pip install ffmpeg-python
        '''
    except ffmpeg.Error as e:
        logging.error(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio, model_size='small', device='cpu'):
    '''
    audio: Audio extracted by extract_audio_ffmpeg()
    model_size: 'tiny' or 'small' or 'large-v3'. Default 'small'
    device: 'cpu' or 'cuda'.  Default 'cpu'
    '''
    try:
        # # model_size = 'large-v3'
        # model_size = input('Input the model size ("tiny" or "large-v3"): ')
        # device = input('Input the device ("cpu" or "cuda"): ')
        model = WhisperModel(model_size_or_path=model_size, device=device, compute_type="int8")
        segments, info = model.transcribe(audio, word_timestamps=True, beam_size=5)
        language = info.language
        language_probability= info.language_probability
        print("Detected language '%s' with probability %f" % (language, language_probability))
        logging.info(f"Transcription language: {language}")
        segments = list(segments)
        for segment in segments:
            logging.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        return language, segments
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return None, None
    
def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return formatted_time

def generate_subtitle_file(input_video,language, segments):
    subtitle_file = os.path.join(os.path.dirname(input_video), f"sub-{input_video}.{language}.srt")
    os.makedirs(os.path.dirname(subtitle_file), exist_ok=True)
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f"{str(index+1)}\n"
        text += f"{segment_start} --> {segment_end}\n"
        text += f"{segment.text}\n\n"
    with open(subtitle_file, "w") as f:
        f.write(text)
    return subtitle_file

def time_to_seconds(time_obj):
    return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000


def create_subtitle_clips(subtitles, videosize, fontsize=24, font='Arial', color='yellow', highlight_color='red'):
    '''
    subtitles: Subtitles after opening using pysrt.open()
    videosize: VideoFileClip(your_video_path).size
    
    '''
    subtitle_clips = []
    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        video_width, video_height = videosize
        
        '''
        `moviepy` version needs to 1.0.3. For `TextClip` to work run the below line codes
        For colab:
        !pip install moviepy
        !apt install imagemagick
        !apt install libmagick++-dev
        !cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
        
        For local system:
        Run `winget install ImageMagick.Q16-HDRI` in command prompt
        After installation, run the code below to add ImageMagick to your system's PATH.
        from moviepy.config import change_settings
        change_settings({"IMAGEMAGICK_BINARY":"C:\your_path\ImageMagick.Q16-HDRI_7.1.1.43_x64__b3hnabsze9y3j\magick.exe"})
        '''

        # Main subtitle text clip
        text_clip = TextClip(
            subtitle.text, fontsize=fontsize, font=font, color=color, bg_color='transparent',
            size=(video_width * 3 / 4, None), method='caption'
        ).set_start(start_time).set_duration(duration)

        # Highlighted text clip
        highlighted_text_clip = TextClip(
            subtitle.text, fontsize=fontsize, font=font, color=highlight_color, bg_color='transparent',
            size=(video_width * 3 / 4, None), method='caption'
        ).set_start(start_time).set_duration(duration)

        subtitle_x_position = 'center'
        subtitle_y_position = video_height * 4 / 5
        text_position = (subtitle_x_position, subtitle_y_position)

        # Set positions
        text_clip = text_clip.set_position(text_position)
        highlighted_text_clip = highlighted_text_clip.set_position(text_position)

        # Append both clips
        subtitle_clips.append(highlighted_text_clip)
        subtitle_clips.append(text_clip)

    return subtitle_clips

def add_subtitle_to_video(video_file, subtitle_file, audio_file):
    video = VideoFileClip(video_file)
    subtitles = pysrt.open(subtitle_file)
    output_video_file = video_file.replace('.mp4', '_subtitled.mp4')

    subtitle_clips = create_subtitle_clips(subtitles, video.size)
    final_video = CompositeVideoClip([video] + subtitle_clips)

    # Add extracted audio to the final video
    audio_clip = AudioFileClip(audio_file)
    final_video = final_video.set_audio(audio_clip)

    final_video.write_videofile(output_video_file)
    return output_video_file

def enhance_video_with_aspect_ratio(input_video, output_video, width=None, height=None):
    try:
        # Load the video
        video = mp.VideoFileClip(input_video)
        
        # Define the desired aspect ratio for TikTok/Shorts (9:16)
        desired_aspect_ratio = 9 / 16
        
        # Get current dimensions
        width, height = video.size
        current_aspect_ratio = width / height
        
        if current_aspect_ratio > desired_aspect_ratio:
            # Width is too large, resize by width and pad top and bottom
            new_width = 720
            new_height = int(new_width / desired_aspect_ratio)
            video = video.resize(width=new_width)
            video = video.margin(top=(new_height - height) // 2, bottom=(new_height - height) // 2)
        else:
            # Height is too large, resize by height and pad left and right
            new_height = 1280
            new_width = int(new_height * desired_aspect_ratio)
            video = video.resize(height=new_height)
            video = video.margin(left=(new_width - width) // 2, right=(new_width - width) // 2)

        # Define output video path
        output_video = os.path.splitext(input_video)[0] + "_aspect_ratio.mp4"
        
        # Write the video file with audio
        video.write_videofile(output_video)

        logging.info(f"Video enhanced for TikTok/Shorts: {output_video}")
        return output_video
    except Exception as e:
        logging.error(f"Error enhancing video: {e}")
        return None

# Process each video in the given folder
def process_videos_in_folder(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_scores = {}
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        frames = extract_frames(video_path)
        predictions = predict_actions(frames)
        quality_metrics = assess_video_quality(frames)
        virality_score = calculate_virality(predictions, quality_metrics)
        video_scores[video_file] = virality_score
    
    scores = list(video_scores.values())
    normalized_scores = normalize_scores(scores)
    
    for i, video_file in enumerate(video_scores):
        video_scores[video_file] = normalized_scores[i]
    
    sorted_video_scores = sorted(video_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_video_scores

# Main function to process video
def main(MODEL_PATH, video_path, model_size, device, output_dir="clips", num_clips=10, clip_length=15):
    audio, sr = extract_audio(video_path)
    loudest_times = audio_detection(audio, sr, num_clips=num_clips, clip_length=clip_length)
    clips = segment_video(video_path, loudest_times, segment_duration=clip_length)
    save_clips(clips, output_dir)
    clip_scores = process_videos_in_folder(output_dir)

    for clip_path, score in clip_scores:
        extracted_audio = extract_audio_ffmpeg(os.path.join(output_dir, clip_path))
        if extracted_audio:
            language, segments = transcribe_audio(extracted_audio, model_size, device)
            if language and segments:
                enhanced_video = enhance_video_with_aspect_ratio(os.path.join(output_dir, clip_path), os.path.join(output_dir, f"enhanced-{clip_path}"), width=1280)
                if enhanced_video:
                    subtitle_file = generate_subtitle_file(input_video=enhanced_video,language=language, segments=segments)
                    if subtitle_file:
                        add_subtitle_to_video(video_file=enhanced_video, subtitle_file=subtitle_file, audio_file=extracted_audio)
    for clip_path, score in clip_scores:  
        logging.info(f"Clip: {clip_path}, Virality Score: {score}")

if __name__ == "__main__":
    MODEL_PATH = input('Input model path: ')
    video_path = input('Input video path: ') # Adjust path to your video
    model_size = input("Input model size ('tiny' or 'large-v3'): ")
    device =     input("Input device ('cpu' or 'cuda'): ")
    main(MODEL_PATH,video_path, model_size, device)

