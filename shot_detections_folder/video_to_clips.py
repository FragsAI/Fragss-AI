import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
import librosa

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    audio, sr = librosa.load(audio_path, sr=None)
    os.remove(audio_path)
    return audio, sr

def find_loudest_moments(audio, sr, num_clips=15, clip_length=5):
    frame_length = sr * clip_length
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=frame_length)[0]
    loudest_indices = np.argsort(rms)[-num_clips:]
    loudest_times = loudest_indices * clip_length
    return loudest_times

def save_clips(video_path, times, output_dir, clip_length=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_length = int(fps * clip_length)
    
    for i, start_time in enumerate(times):
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        out = cv2.VideoWriter(f"{output_dir}/clip_{i+1}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frames_written = 0
        while frames_written < frame_length:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1

        out.release()

    cap.release()

def main(video_path, output_dir="clips", num_clips=15, clip_length=5):
    audio, sr = extract_audio(video_path)
    loudest_times = find_loudest_moments(audio, sr, num_clips=num_clips, clip_length=clip_length)
    save_clips(video_path, loudest_times, output_dir, clip_length=clip_length)

if __name__ == "__main__":
    video_path = '/Users/kesinishivaram/FragsAI/Fragss-AI/cod.mp4'
    main(video_path)
