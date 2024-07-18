import pandas as pd
from pytube import YouTube
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load CSV file
df = pd.read_csv('/Users/kesinishivaram/FragsAI/valorant_shorts.csv')

# Function to download video
def download_video(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        audio = yt.streams.get_audio_only()
        if video:
            video.download(output_path='downloads/', filename=f"{video_id}.mp4")
            audio.download(output_path='downloads/', filename=f"{video_id}.mp3")
            print(f"Downloaded: {video_id}")
        else:
            print(f"No suitable stream found for: {video_id}")
    except Exception as e:
        print(f"Failed to download {video_id}: {e}")

# Iterate over video IDs and download
for video_id in df['id']:
    download_video(video_id)
