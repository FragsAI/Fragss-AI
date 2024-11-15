import os
import math
import ffmpeg
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from faster_whisper import WhisperModel

input_video = "/Users/kesinishivaram/FragsAI/clips/clip_1.mp4"
input_video_name = os.path.splitext(os.path.basename(input_video))[0]

def extract_audio():
    extracted_audio = os.path.join(os.path.dirname(input_video), f"audio-{input_video_name}.wav")
    os.makedirs(os.path.dirname(extracted_audio), exist_ok=True)
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio

def transcribe(audio):
    model = WhisperModel("small")
    segments, info = model.transcribe(audio)
    language = info[0]
    print("Transcription language", info[0])
    segments = list(segments)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return language, segments

def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return formatted_time

def generate_subtitle_file(language, segments):
    subtitle_file = os.path.join(os.path.dirname(input_video), f"sub-{input_video_name}.{language}.srt")
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

def create_subtitle_clips(subtitles, videosize, fontsize=24, font='Arial', color='yellow'):
    subtitle_clips = []
    for subtitle in subtitles:
        start_time = time_to_seconds(subtitle.start)
        end_time = time_to_seconds(subtitle.end)
        duration = end_time - start_time

        video_width, video_height = videosize
        text_clip = TextClip(
            subtitle.text, fontsize=fontsize, font=font, color=color, bg_color='black',
            size=(video_width * 3 / 4, None), method='caption'
        ).set_start(start_time).set_duration(duration)

        subtitle_x_position = 'center'
        subtitle_y_position = video_height * 4 / 5
        text_position = (subtitle_x_position, subtitle_y_position)
        subtitle_clips.append(text_clip.set_position(text_position))

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

def run():
    extracted_audio = extract_audio()
    language, segments = transcribe(audio=extracted_audio)
    subtitle_file = generate_subtitle_file(language=language, segments=segments)
    add_subtitle_to_video(video_file=input_video, subtitle_file=subtitle_file, audio_file=extracted_audio)

if __name__ == "__main__":
    run()
