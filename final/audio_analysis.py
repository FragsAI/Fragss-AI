
import os
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import logging
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as mtf
from pydub import AudioSegment
from pydub.silence import detect_silence
import speech_recognition as sr
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Extract audio from video
def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    """
    Extracts audio from a video file.
    Args:
        video_path (str): Path to the video file.
        output_audio_path (str): Path to save the extracted audio.
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    return output_audio_path

# Analyze audio for gunshot sounds
def analyze_gunshots(audio_path):
    """
    Analyzes the audio file to detect gunshot events.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        list: Timestamps of detected gunshot events.
    """
    y, sr = librosa.load(audio_path, sr=None)
    S = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(S=S).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

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

# Analyze audio for laughter
def analyze_laughter(audio_path):
    """
    Analyzes the audio file to detect laughter events.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        list: Timestamps of detected laughter events.
    """
    mt_feats, st_feats, _ = mtf.mid_feature_extraction_to_file(audio_path, "temp_features", 1.0, 1.0, 0.05, 0.05, True, False)
    model_path = "laughterSVM"
    class_names, mt_win, mt_step, st_win, st_step, classifier, mean, std, class_names = aT.load_model(model_path)

    [Result, P, classNames] = aT.file_classification(audio_path, model_path, "svm", False)

    laughter_threshold = 0.5
    laughter_events = []
    for i in range(len(Result)):
        if Result[i] == 1 and P[i][1] > laughter_threshold:
            laughter_events.append(i * mt_step)

    return laughter_events

# Transcribe audio with timestamps
def transcribe_audio_with_timestamps(audio_path):
    """
    Transcribes audio with timestamps using silence detection.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        list: Transcriptions with timestamps.
    """
    audio = AudioSegment.from_wav(audio_path)
    silence_segments = detect_silence(audio, silence_thresh=-50, min_silence_len=500)

    recognizer = sr.Recognizer()
    transcription_with_timestamps = []

    for start, end in tqdm(silence_segments, desc="Transcribing audio"):
        with sr.AudioFile(audio_path) as source:
            source_offset = start / 1000.0
            source_duration = (end - start) / 1000.0
            audio_data = recognizer.record(source, offset=source_offset, duration=source_duration)

            try:
                text = recognizer.recognize_google(audio_data)
                timestamp_str = f"[{start / 1000:.2f} - {end / 1000:.2f}]"
                transcription_with_timestamps.append(f"{timestamp_str} {text}")
            except sr.UnknownValueError:
                transcription_with_timestamps.append(f"[{start / 1000:.2f} - {end / 1000:.2f}] [Unintelligible]")

    return transcription_with_timestamps

# Audio analysis pipeline
def audio_analysis_pipeline(video_path):
    """
    Complete audio analysis pipeline for gunshot detection, laughter analysis, and transcription.
    Args:
        video_path (str): Path to the video file.
    Returns:
        dict: Analysis results including gunshot events, laughter events, and transcription.
    """
    # Extract audio from video
    audio_path = extract_audio(video_path)

    # Gunshot detection
    gunshot_events = analyze_gunshots(audio_path)
    logging.info(f"Detected gunshot events at: {gunshot_events}")

    # Laughter detection
    laughter_events = analyze_laughter(audio_path)
    logging.info(f"Detected laughter events at: {laughter_events}")

    # Transcription with timestamps
    transcription = transcribe_audio_with_timestamps(audio_path)
    logging.info(f"Transcription completed with timestamps.")

    # Combine all results into a dictionary
    audio_analysis_results = {
        "gunshot_events": gunshot_events,
        "laughter_events": laughter_events,
        "transcription": transcription
    }

    return audio_analysis_results
