import os
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence

def transcribe_video(video_path, output_dir="transcription", silence_thresh=-50, min_silence_len=500):
    """
    Transcribes audio from a video file.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save transcriptions.
        silence_thresh (int): Silence threshold for splitting audio.
        min_silence_len (int): Minimum silence length.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "audio.wav")

        # Extract audio from the video
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, logger=None)
        
        # Split audio on silence
        audio = AudioSegment.from_wav(audio_path)
        silence_segments = split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)

        recognizer = sr.Recognizer()
        transcription_with_timestamps = []
        
        for start, end in silence_segments:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source, offset=start, duration=(end - start))
                try:
                    text = recognizer.recognize_google(audio_data)
                    transcription_with_timestamps.append(f"[{start} - {end}]: {text}")
                except sr.UnknownValueError:
                    transcription_with_timestamps.append(f"[{start} - {end}]: [Unintelligible]")
        
        output_file_path = os.path.join(output_dir, "transcription.txt")
        with open(output_file_path, 'w') as file:
            file.write('\n'.join(transcription_with_timestamps))
        print(f"Transcription saved to: {output_file_path}")
    
    except Exception as e:
        print(f"Error during transcription: {e}")
