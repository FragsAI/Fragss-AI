import os
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
import requests
import json
import ast
import re
import torch
from pathlib import Path 
import os
import gc
import numpy as np

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def edit_paths(file_path):
    # Extract filename and extension
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    ext = ext.lstrip('.').lower()  # Remove dot and make lowercase

    # Validate allowed extensions
    valid_exts = {'mp4', 'avi', 'mp3', 'wav'}
    if ext not in valid_exts:
        raise ValueError(f"Unsupported file extension: .{ext}")

    # Generate new file path
    dir_name = os.path.dirname(file_path)
    edited_name = f"{name}_edited_output.{ext}"
    edited_path = os.path.join(dir_name, edited_name)

    count = 1
    while os.path.exists(edited_path):
        edited_name = f"{name}_edited_output{count}.{ext}"
        edited_path = os.path.join(dir_name, edited_name)
        count += 1

    return edited_path
   
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name= torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

# Step 1: Transcribe the Video
def transcribe_video(video_path, model_name='tiny', device='cuda'):
    video_name = [ i for i in video_path.split('/') if i.endswith(('.mp4','.avi'))][0][:-4]
    audio_path = f"{video_name}_audio_temp.wav"
    if os.path.exists(audio_path)==False:  
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
    else:
        logging.info(f"File '{audio_path}' already exists. Not overwriting")
    
    if os.path.exists(f"{video_name}_transcription_temp.npy")==False:
        model = whisper.load_model(model_name, device)
        audio_path = f"{video_name}_audio_temp.wav"
        os.system(f"ffmpeg -i {video_path} -ar 16000 -ac 1 -b:a 64k -f mp3 {audio_path}")
        result = model.transcribe(audio_path)
        transcription = []
        for segment in result['segments']:
            transcription.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        np.save(f"{video_name}_transcription_temp",np.array(transcription)) 
        return transcription
    else:
        logging.info(f"File '{video_name}_transcription_temp.npy' already exists. Not transcribing.")
        allow_pickle=True
        transcription=list(np.load(file=f"{video_name}_transcription_temp.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII', max_header_size=10000))
        return transcription
         
def get_relevant_segments(transcript, user_query):
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY" # Replace with your API key
    }

    data = {
        "messages": [
            {
                "role": "system",
                "content": prompt
            }
        ],
        "model": "llama3-8b-8192",
        "temperature": 1,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False,
        "stop": None
    }
    response = requests.post(url, headers=headers, json=data)
    data = response.json()["choices"][0]["message"]["content"]

    # Extract the JSON-like part from the string
    match = re.search(r'\{.*"conversations".*\}', data, re.DOTALL)
    if match:
        json_string = match.group(0)
        # Convert to dictionary safely
        conversations = ast.literal_eval(json_string)["conversations"]
        for conv in conversations:
            conv['start'] = float(conv['start'])
            conv['end'] = float(conv['end'])
        logging.info(f"{len(conversations)} relevant segments found!")
        return conversations
    else:
        logging.info("No valid JSON structure found.")

def split_transcript(transcript, max_chunk_token_size=3500):
    chunks = []
    current_chunk = []
    current_token_est = 0

    for segment in transcript:
        segment_text = f"{segment}"
        segment_token_est = len(segment_text) // 4  # rough token estimate

        if current_token_est + segment_token_est > max_chunk_token_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_est = 0

        current_chunk.append(segment)
        current_token_est += segment_token_est

    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def edit_video(original_video_path, segments, fade_duration=0.5):
    output_video_path=edit_paths(original_video_path)
    video = VideoFileClip(original_video_path)
    clips = []
    for seg in segments:
        start = seg['start']
        end = seg['end']
        # clip = video.subclip(start, end).fadein(fade_duration).fadeout(fade_duration)
        # clips.append(clip)
        if start is not None and end is not None and start < end:
            clip = video.subclip(start, end).fadein(fade_duration).fadeout(fade_duration)
            clips.append(clip)
        else:
            # logging.info(f"Skipping invalid segment: {seg}")
            pass
            
    if clips:
        logging.info(f"Concatenating {len(clips)} sublips")
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        # If type error occurs the add below line of codes after lin no. 299 in VideoClip.py of moviepy library in video folder
        # if fps is None:
        #     fps = self.fps
    else:
        logging.info("No segments to include in the edited video.")

def main():
    # # Paths
    # input_video = "animals.mp4"
    # # User Query
    # user_query = "Find the panda climbing down"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name= torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    # Step 1: Transcribe
    logging.info("Transcribing video...")
    model_name="tiny"
    transcription = transcribe_video(input_video, model_name=model_name, device=device)
    logging.info("Done!\n")
    
    logging.info("Getting relevant segments...")
    # relevant_segments = get_relevant_segments(transcription, user_query)
    chunks = split_transcript(transcription)
    # logging.info(f"Total chunks: {len(chunks)}")
    for chunk in chunks:
        relevant_segments = get_relevant_segments(chunk, user_query)
    logging.info("Done!\n")
    
    # Step 5: Edit Video
    if relevant_segments != None:
        logging.info("Editing video...")
        edit_video(input_video, relevant_segments)
        logging.info("Done!")
    else:
        logging.info("No relevant segments to edit.")
        

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name= torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        logging.info(f"Available GPU: {device_name}\n")
        with torch.device(device):
            main()
    else:
        logging.info("No GPU available!\n")
        main()  
