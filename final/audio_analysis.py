import os
import logging
import numpy as np
import librosa
import speech_recognition as sr
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_silence
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as mtf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    """
    Extracts audio from a video file.
    """
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path)
        return output_audio_path
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None

def analyze_gunshots(audio_path, threshold=0.2, min_duration=1):
    """
    Detects potential gunshot events based on audio energy levels.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y).flatten()
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        gunshot_events = [t for t, r in zip(times, rms) if r > threshold]
        filtered_events = []
        last_event = -min_duration
        for event in gunshot_events:
            if event - last_event >= min_duration:
                filtered_events.append(event)
                last_event = event
        
        return filtered_events
    except Exception as e:
        logging.error(f"Error analyzing gunshots: {e}")
        return []

def analyze_laughter(audio_path, model_path="laughterSVM"):
    """
    Detects laughter events using a trained SVM model.
    """
    try:
        class_names, _, _, _, _, classifier, mean, std, _ = aT.load_model(model_path)
        result, probabilities, _ = aT.file_classification(audio_path, model_path, "svm", False)
        laughter_events = [i for i, p in enumerate(probabilities) if result[i] == 1 and p[1] > 0.5]
        return laughter_events
    except Exception as e:
        logging.error(f"Error analyzing laughter: {e}")
        return []

def transcribe_audio_with_timestamps(audio_path):
    """
    Transcribes speech from an audio file, marking timestamps based on silence detection.
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
        silence_segments = detect_silence(audio, silence_thresh=-50, min_silence_len=500)
        recognizer = sr.Recognizer()
        transcription = []
        
        for start, end in tqdm(silence_segments, desc="Transcribing audio"):
            with sr.AudioFile(audio_path) as source:
                try:
                    offset, duration = start / 1000.0, (end - start) / 1000.0
                    audio_data = recognizer.record(source, offset=offset, duration=duration)
                    text = recognizer.recognize_google(audio_data)
                    transcription.append(f"[{offset:.2f} - {end/1000:.2f}] {text}")
                except sr.UnknownValueError:
                    transcription.append(f"[{offset:.2f} - {end/1000:.2f}] [Unintelligible]")
        
        return transcription
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return []

def audio_analysis_pipeline(video_path):
    """
    Processes video audio for gunshot detection, laughter detection, and transcription.
    """
    audio_path = extract_audio(video_path)
    if not audio_path:
        return {}
    
    gunshot_events = analyze_gunshots(audio_path)
    laughter_events = analyze_laughter(audio_path)
    transcription = transcribe_audio_with_timestamps(audio_path)
    
    logging.info(f"Gunshot events: {gunshot_events}")
    logging.info(f"Laughter events: {laughter_events}")
    logging.info(f"Transcription complete.")
    
    return {
        "gunshot_events": gunshot_events,
        "laughter_events": laughter_events,
        "transcription": transcription
    }
