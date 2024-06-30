import os
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import moviepy.editor as mp
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as mtf

# Path to the video file
video_path = "input_video.mp4"
output_dir = "output_clips"

# Step 1: Extract audio from video
def extract_audio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)

# Step 2: Analyze audio for gunshot sounds
def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=S).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    
    # Gunshot detection: simple thresholding
    threshold = 0.2
    min_duration_between_gunshots = 1
    
    gunshot_events = times[rms > threshold]
    
    filtered_events = []
    last_event = -min_duration_between_gunshots
    for event in gunshot_events:
        if event - last_event >= min_duration_between_gunshots:
            filtered_events.append(event)
            last_event = event
    
    return filtered_events

# Step 3: Analyze audio for laughter (funny moments)
def analyze_laughter(audio_path):
    # Extract mid-term features and save them to a temporary file
    mt_feats, st_feats, _ = mtf.mid_feature_extraction_to_file(audio_path, "temp_features", 1.0, 1.0, 0.05, 0.05, True, False)
    
    # Load pre-trained SVM laughter model
    model_path = "laughterSVM"
    class_names, mt_win, mt_step, st_win, st_step, classifier, mean, std, class_names = aT.load_model(model_path)
    
    # Predict laughter segments
    [Result, P, classNames] = aT.file_classification(audio_path, model_path, "svm", False)
    
    # Threshold based on the laughter probability
    laughter_threshold = 0.5
    laughter_events = []
    for i in range(len(Result)):
        if Result[i] == 1 and P[i][1] > laughter_threshold:
            laughter_events.append(i * mt_step)
    
    return laughter_events

# Step 4: Segment video based on audio analysis
def segment_video(video_path, events, segment_duration=5):
    video = VideoFileClip(video_path)
    clips = []
    
    for event in events:
        start_time = max(0, event - segment_duration / 2)
        end_time = min(video.duration, event + segment_duration / 2)
        clip = video.subclip(start_time, end_time)
        clips.append(clip)
    
    return clips

# Step 5: Save video clips
def save_clips(clips, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, clip in enumerate(clips):
        output_path = os.path.join(output_dir, f"clip_{i + 1}.mp4")
        clip.write_videofile(output_path)

def main():
    audio_path = "extracted_audio.wav"
    
    # Extract audio
    extract_audio(video_path, audio_path)
    
    # Analyze audio for gunshot sounds
    gunshot_events = analyze_audio(audio_path)
    
    # Analyze audio for laughter
    laughter_events = analyze_laughter(audio_path)
    
    # Combine events
    events = sorted(set(gunshot_events + laughter_events))
    
    # Segment video
    clips = segment_video(video_path, events)
    
    # Save clips
    save_clips(clips, output_dir)

if __name__ == "__main__":
    main()
