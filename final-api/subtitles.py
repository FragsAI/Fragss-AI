import os
import math
import ffmpeg
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from faster_whisper import WhisperModel


def extract_audio(video_path):
    """
    Extracts audio from a video file.
    Args:
        video_path (str): Path to the video file.
    Returns:
        str: Path to the extracted audio file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted_audio = os.path.join(os.path.dirname(video_path), f"audio-{video_name}.wav")
    os.makedirs(os.path.dirname(extracted_audio), exist_ok=True)
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio


def transcribe(audio_path):
    """
    Transcribes audio using the Whisper model.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        tuple: Transcription language and segments.
    """
    model = WhisperModel("small")
    segments, info = model.transcribe(audio_path)
    language = info[0]
    print("Transcription language:", language)
    return language, segments


def format_time(seconds):
    """
    Formats seconds into SRT time format (hh:mm:ss,ms).
    """
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_subtitle_file(language, segments, video_path):
    """
    Generates a subtitle file (SRT) from transcription segments.
    Args:
        language (str): Detected language of the audio.
        segments (list): List of transcription segments.
        video_path (str): Path to the video file.
    Returns:
        str: Path to the generated subtitle file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    subtitle_file = os.path.join(os.path.dirname(video_path), f"sub-{video_name}.{language}.srt")
    os.makedirs(os.path.dirname(subtitle_file), exist_ok=True)
    with open(subtitle_file, "w") as f:
        for index, segment in enumerate(segments):
            start = format_time(segment.start)
            end = format_time(segment.end)
            f.write(f"{index+1}\n{start} --> {end}\n{segment.text}\n\n")
    return subtitle_file


def time_to_seconds(time_obj):
    """
    Converts a pysrt time object to seconds.
    """
    return time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds + time_obj.milliseconds / 1000


def create_subtitle_clips(subtitles, videosize, fontsize=24, font='Arial', color='yellow'):
    """
    Creates subtitle clips to overlay on a video.
    """
    subtitle_clips = []
    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        video_width, video_height = videosize
        text_clip = TextClip(
            subtitle.text, fontsize=fontsize, font=font, color=color,
            bg_color='black', size=(video_width * 3 / 4, None), method='caption'
        ).set_start(start_time).set_duration(duration)

        subtitle_clips.append(text_clip.set_position(("center", video_height * 4 / 5)))

    return subtitle_clips


def add_subtitle_to_video(video_path, subtitle_file, audio_file, font, color):
    """
    Adds subtitles to a video with custom font and color.
    Args:
        video_path (str): Path to the video file.
        subtitle_file (str): Path to the subtitle file.
        audio_file (str): Path to the audio file.
        font (str): Font of the subtitles.
        color (str): Color of the subtitles.
    Returns:
        str: Path to the subtitled video file.
    """
    video = VideoFileClip(video_path)
    subtitles = pysrt.open(subtitle_file)
    output_video_file = video_path.replace('.mp4', '_subtitled.mp4')

    subtitle_clips = create_subtitle_clips(subtitles, video.size, font=font, color=color)
    final_video = CompositeVideoClip([video] + subtitle_clips)

    # Add extracted audio to the final video
    audio_clip = AudioFileClip(audio_file)
    final_video = final_video.set_audio(audio_clip)

    final_video.write_videofile(output_video_file, codec="libx264")
    return output_video_file


def apply_subtitles_to_clips(clip_paths, font="Arial", color="yellow"):
    """
    Applies subtitles to a list of video clips.
    Args:
        clip_paths (list): List of video clip paths.
        font (str): Font of the subtitles.
        color (str): Color of the subtitles.
    """
    for clip_path in clip_paths:
        extracted_audio = extract_audio(clip_path)
        language, segments = transcribe(extracted_audio)
        subtitle_file = generate_subtitle_file(language, segments, clip_path)
        add_subtitle_to_video(clip_path, subtitle_file, extracted_audio, font, color)
