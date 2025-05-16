# Import dependencies
import os
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
import requests
import json
import ast
import re
import torch
from pathlib import Path 
import gc
import numpy as np
import logging

# Configure logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility function
def edit_paths(file_path):
    """
    Generates a unique output path for the edited video/audio by appending '_edited_output'.

    Args:
        file_path (str): Original file path.

    Returns:
        str: New unique output path.
    """
    base_name = os.path.basename(file_path)  # Extract file name from path
    name, ext = os.path.splitext(base_name)  # Split name and extension
    ext = ext.lstrip('.').lower()  # Clean extension

    valid_exts = {'mp4', 'avi', 'mp3', 'wav'}
    if ext not in valid_exts:
        raise ValueError(f"Unsupported file extension: .{ext}")

    dir_name = os.path.dirname(file_path)  # Extract directory name
    edited_name = f"{name}_edited_output.{ext}"  # Create new file name
    edited_path = os.path.join(dir_name, edited_name) # Join new file name with directory

    count = 1
    while os.path.exists(edited_path):  # Ensure filename uniqueness
        edited_name = f"{name}_edited_output{count}.{ext}"
        edited_path = os.path.join(dir_name, edited_name)
        count += 1

    return edited_path

# Step 1: Transcribe video
def transcribe_video(video_path, model_name='tiny', device='cuda'):
    """
    Transcribes audio from the video using Whisper and stores transcript.

    Args:
        video_path (str): Path to video file.
        model_name (str): Whisper model name, e.g. 'tiny'. Other model names- 'base' 'small', 'medium', 'large', 'turbo').
        device (str): 'cuda' or 'cpu'.

    Returns:
        list: Transcribed segments with start, end, and text.
    """
    video_name = [i for i in video_path.split('/') if i.endswith(('.mp4','.avi'))][0][:-4]  # Extract base name without extension
    audio_path = f"{video_name}_audio_temp.wav"  # Temporary audio file path

    # Extract audio if not already extracted
    if not os.path.exists(audio_path):  
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
    else:
        logging.info(f"File '{audio_path}' already exists. Not overwriting")

    # Transcribe step    
    transcription_file = f"{video_name}_transcription_temp.npy"  # Temp transcription file name
    if not os.path.exists(transcription_file):  # Only transcribe if not already done
        model = whisper.load_model(model_name, device)
        os.system(f"ffmpeg -i {video_path} -ar 16000 -ac 1 -b:a 64k -f mp3 {audio_path}")  # Convert audio format
        result = model.transcribe(audio_path) # transcribe
        
        # Format the transcription output by extracting and cleaning up start time, end time, and text for each segment
        transcription = [
            {"start": segment['start'], "end": segment['end'], "text": segment['text'].strip()}
            for segment in result['segments']
        ]
        np.save(transcription_file, np.array(transcription))  # Save transcript
        return transcription
    else:
        logging.info(f"File '{transcription_file}' already exists. Not transcribing.")
        return list(np.load(transcription_file, allow_pickle=True))  # Load existing transcription

# Step 2: Split transcript
def split_transcript(transcript, max_chunk_token_size=3500):
    """
    Splits transcript into chunks to respect token limits.

    Args:
        transcript (list): List of transcript segments.
        max_chunk_token_size (int): Maximum tokens per chunk.

    Returns:
        list: List of transcript chunks.
    """
    chunks = []  # List to store final chunks
    current_chunk = []  # Buffer for current chunk
    current_token_est = 0  # Track token estimate

    for segment in transcript:
        segment_text = f"{segment}"  # Convert to string
        segment_token_est = len(segment_text) // 4  # Approx token count

        if current_token_est + segment_token_est > max_chunk_token_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_est = 0

        current_chunk.append(segment)
        current_token_est += segment_token_est

    if current_chunk:  # Add remaining chunk
        chunks.append(current_chunk)

    return chunks

# Step 3: Get relevant segments
def get_relevant_segments(transcript, user_query):
    """
    Uses Groq API to find video transcript segments relevant to a user query.

    Args:
        transcript (list): Transcript segments.
        user_query (str): Search phrase.

    Returns:
        list: List of relevant segment start-end times.
    """
    prompt = f"""You are an expert video editor who can read video transcripts and perform video editing. Given a transcript with segments, your task is to identify all the conversations related to a user query. Follow these guidelines when choosing conversations. A group of continuous segments in the transcript is a conversation.

Guidelines:
1. The conversation should be relevant to the user query. The conversation should include more than one segment to provide context and continuity.
2. Include all the before and after segments needed in a conversation to make it complete.
3. The conversation should not cut off in the middle of a sentence or idea.
4. Choose multiple conversations from the transcript that are relevant to the user query.
5. Match the start and end time of the conversations using the segment timestamps from the transcript.
6. The conversations should be a direct part of the video and should not be out of context.

Output format: {{ "conversations": [{{"start": "s1", "end": "e1"}}, {{"start": "s2", "end": "e2"}}] }}

Transcript:
{transcript}

User query:
{user_query}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    YOUR_API_KEY='your_api_key' # Replace with your API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {YOUR_API_KEY}"  # Replace with your API key
    }

    data = {
        "messages": [
            {"role": "system", "content": prompt}
        ],
        "model": "llama3-8b-8192",
        "temperature": 1,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None
    }
    
    # Send a POST request to the Groq API with the prompt and input data to retrieve relevant transcript segments
    response = requests.post(url, headers=headers, json=data) 
    data = response.json()["choices"][0]["message"]["content"]  # Extract response content

    match = re.search(r'\{.*"conversations".*\}', data, re.DOTALL)  # Find conversation JSON
    if match:
        conversations = ast.literal_eval(match.group(0))["conversations"]  # Parse JSON safely
        for conv in conversations:
            conv['start'] = float(conv['start'])
            conv['end'] = float(conv['end'])
        logging.info(f"{len(conversations)} relevant segments found!")
        return conversations
    else:
        logging.info("No valid JSON structure found.")
        return None

# Step 4: Edit video
def edit_video(original_video_path, segments, fade_duration=0.5):
    """
    Extracts and concatenates video clips from relevant segments.

    Args:
        original_video_path (str): Path to original video.
        segments (list): List of {start, end} dicts.
        fade_duration (float): Duration of fadein/fadeout in seconds.
    """
    output_video_path = edit_paths(original_video_path)  # Create output path
    video = VideoFileClip(original_video_path)  # Load video
    clips = []  # Store relevant subclips

    for seg in segments:
        start, end = seg['start'], seg['end']
        if start is not None and end is not None and start < end:
            clip = video.subclip(start, end).fadein(fade_duration).fadeout(fade_duration)  # Extract subclip with fades
            clips.append(clip)

    if clips:
        logging.info(f"Concatenating {len(clips)} subclips")
        final_clip = concatenate_videoclips(clips, method="compose")  # Concatenate all clips
        final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")  # Export video
        # IMPORTANT: If type error occurs the add below line of codes after line no. 299 in VideoClip.py of moviepy library in video folder
        # if fps is None:
        #     fps = self.fps
    else:
        logging.info("No segments to include in the edited video.")

# Main pipeline
def main():
    """
    Pipeline for transcribing video, extracting relevant segments, and editing output video.
    """
    input_video = "your_video_file.mp4"  # Replace with your input video file path
    user_query = "your prompt"  # Replace with your prompt(s)

    # Step 1: Transcribe video
    logging.info("Transcribing video...")
    transcription = transcribe_video(input_video, model_name="tiny", device=device) 
    
    # Step2: Splitting transcript into chunks
    logging.info("Splitting transcript into chunks...")
    chunks = split_transcript(transcription)  # Chunk transcript
    
    # Step 3: Get relevant segments
    all_segments = []
    for chunk in chunks:
        relevant_segments = get_relevant_segments(chunk, user_query)  
        if relevant_segments:
            all_segments.extend(relevant_segments)

    # Step 4: Edit video       
    logging.info("Editing video...")
    if all_segments:
        edit_video(input_video, all_segments)  # Generate final edited video
        logging.info("Done!")
    else:
        logging.info("No relevant segments to edit.")

if __name__ == "__main__":
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Empty CUDA cache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    logging.info(f"Device in use: {device_name}")
    main()
