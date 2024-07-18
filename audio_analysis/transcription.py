import os
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence

def detect_silence(audio, silence_thresh=-50, min_silence_len=500):
    silence_segments = []
    non_silence_segments = split_on_silence(
        audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len
    )
    start_time = 0

    for segment in non_silence_segments:
        end_time = start_time + segment.duration_seconds
        silence_segments.append((start_time, end_time))
        start_time = end_time

    return silence_segments

def transcribe_video(video_path):
    try:
        # Ensure the output directory exists
        audio_output_dir = "transcription"
        os.makedirs(audio_output_dir, exist_ok=True)
        audio_path = os.path.join(audio_output_dir, "audio.wav")

        # Extract audio from the video
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path, logger=None)
        
        # Use pydub to split the audio on silence and get the timestamps
        audio = AudioSegment.from_wav(audio_path)
        silence_segments = detect_silence(audio, silence_thresh=-50, min_silence_len=500)

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Transcribe the audio in chunks
        transcription_with_timestamps = []
        
        for start, end in silence_segments:
            with sr.AudioFile(audio_path) as source:
                source_offset = start * 1000
                source_duration = (end - start) * 1000
                audio_data = recognizer.record(source, offset=source_offset, duration=source_duration)
                
                try:
                    text = recognizer.recognize_google(audio_data)
                    timestamp_str = f"[{start:.2f} - {end:.2f}]"
                    transcription_with_timestamps.append(f"{timestamp_str} {text}")
                except sr.UnknownValueError:
                    transcription_with_timestamps.append(f"{timestamp_str} [Unintelligible]")

        # Save the transcription to a text file
        output_file_path = os.path.join(audio_output_dir, "transcription.txt")
        with open(output_file_path, 'w') as file:
            file.write('\n'.join(transcription_with_timestamps))

        print(f"Transcription saved to: {output_file_path}")

    except Exception as e:
        print(f"Error during transcription: {e}")

# Pipeline for video transcription
video_path = "/Users/kesinishivaram/FragsAI/Fragss-AI/cod.mp4"
# Call the transcribe_video function
transcribe_video(video_path)
