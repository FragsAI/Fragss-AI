# Import essential libraries
import supervision as sv  # For video frame extraction and saving images
from moviepy.editor import VideoFileClip, concatenate_videoclips  # For video editing
import moviepy
import cv2  # OpenCV for video I/O
import torch
import torchvision
import numpy as np
import random
import os
import shutil
import psutil
import time
from tqdm.auto import tqdm

# Transformers and related deep learning tools
import flash_attn  # Flash attention support for transformer speedup
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor  # Florence-2 model interface
import timm.layers  # Vision transformer layers

# PIL and visualization
from PIL import Image, ImageDraw, ImageFont  # Image processing and drawing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Text processing
import string
from nltk.corpus import stopwords
import nltk

# Logging and warnings
import math
import logging
import warnings
warnings.filterwarnings('ignore')

# Download stopwords if not already present
# nltk.download('stopwords')

# Check if stopwords are already present before downloading
if not 'stopwords' in nltk.data.find('corpora/stopwords.zip'):
    nltk.download('stopwords')

# Configure logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def adjust_sample_interval(video_path):
    """
    Dynamically adjust frame sampling interval based on video length.
    Args:
      video_path (str): Path to the input video file.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Video duration in seconds
    cap.release()

    # Adjust sampling based on video duration
    if duration <= 3600:  # 1 hour
        return 10  # Sample every 10th frame
    elif duration <= 18000:  # 5 hours
        return 15  # Sample every 15th frame
    else:
        return 20  # Sample every 20th frame

def determine_chunk_size():
    """ Adjust chunk size dynamically based on available memory. """
    available_memory = psutil.virtual_memory().available  # Get available RAM in bytes

    if available_memory < 4 * 1024**3:  # Less than 4GB RAM
        return 500  # Small chunks
    elif available_memory < 8 * 1024**3:  # Less than 8GB RAM
        return 1000
    elif available_memory < 16 * 1024**3:  # Less than 16GB RAM
        return 2000
    else:
        return 5000  # Large chunks for high-memory systems

def save_frames_and_indices_in_batches(video_path, total_frames, batch_size, base_dir, sample_interval=15):
    """
    Extracts and saves sampled frames from a video into batch directories.
    Also saves their indices for later retrieval and alignment.

    Args:
      video_path (str): Path to the input video file.
      total_frames (int): The total number of frames in the video to process.
      batch_size (int): The number of frames to save in each batch.
      base_dir (str): The directory where the batches will be saved.
      sample_interval (int, optional): The interval between sampled frames (default is 15).
                                        Only frames at this interval will be saved.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    sub_dir=os.path.join(base_dir, f"{video_name}_frames_batches")
    if os.path.exists(sub_dir)==False:
      frames_generator = sv.get_video_frames_generator(video_path)
      frame_idx = 0
      batch_num = 0
      with tqdm(total=total_frames, desc="Extracting frames") as pbar:
          while True:
              batch_dir = os.path.join(sub_dir, f'{video_name}_frames_batch_{batch_num:04d}')
              os.makedirs(batch_dir, exist_ok=True)
              saved_in_batch = 0
              frame_indices = []

              with sv.ImageSink(batch_dir, image_name_pattern="{:05d}.jpeg") as sink:
                  while saved_in_batch < batch_size:
                      try:
                          frame = next(frames_generator)
                      except StopIteration:
                          break

                      # Save frame at given sample interval
                      if frame_idx % sample_interval == 0:
                          sink.save_image(frame, image_name=f"{frame_idx:05d}.jpeg")
                          frame_indices.append(frame_idx)
                          saved_in_batch += 1

                      frame_idx += 1
                      pbar.update(1)

                      if frame_idx >= total_frames:
                          break

              if frame_indices:
                  indices_path = os.path.join(batch_dir, f"{video_name}_frame_indices_batch_{batch_num:04d}.npy")
                  np.save(indices_path, np.array(frame_indices, dtype=np.int32))

              if saved_in_batch == 0 or frame_idx >= total_frames:
                  break

              batch_num += 1
      logging.info((f"Saved up to frame {frame_idx} in {batch_num + 1} batches in {sub_dir}."))
    else:
      # Extract the file name (with extension)
      file_name = os.path.basename(video_path)
      # logging.info(f'Frames for {file_name} already extracted.')
      print(f'Frames for {file_name} already exist in {sub_dir}. Not extracting!')

def fetch_frames_and_indices_from_batches(base_dir, video_path, batches=True):
    """
    Loads saved image frames and their indices from previously stored batches.
    Useful for avoiding reprocessing the video.

    Args:
        base_dir (str): The base directory where the frame batches are stored.
        video_path (str): The path to the video file, used to derive the video name.
        batches (bool): If True, returns frames and indices as lists of lists from each batch folder.
                        If False, returns all frames and indices in a single list.

    Returns:
        tuple: A tuple containing:
            - frames (list of lists or list of PIL.Image): List of frames, either grouped by batch or in a single list.
            - indices (list of lists or list of int): List of indices, either grouped by batch or in a single list.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_batches_dir = os.path.join(base_dir, f"{video_name}_frames_batches")
    all_frames = []
    all_indices = []
    all_frames_by_batch = []
    all_indices_by_batch = []

    frames_batches_list = sorted(os.listdir(frames_batches_dir))
    total_batches = len(frames_batches_list)

    for batch_num, batch in enumerate(tqdm(frames_batches_list, total=total_batches, desc='Processing batches')):
        batch_dir = os.path.join(frames_batches_dir, batch)
        indices_path = os.path.join(batch_dir, f"{video_name}_frame_indices_batch_{batch_num:04d}.npy")

        if not os.path.isfile(indices_path):
            logging.warning(f"{video_name}_frame_indices_batch_{batch_num:04d}.npy not found in {batch_dir}. Skipping.")
            continue

        frame_indices = np.load(indices_path)
        frame_files = sorted([fname for fname in os.listdir(batch_dir) if fname.endswith((".jpeg", ".jpg"))])

        # Log if index count and frame count mismatch
        if len(frame_files) != len(frame_indices):
            logging.info(f"Mismatch: {batch_dir} has {len(frame_files)} frames and {len(frame_indices)} indices.")

        # Load each image and record its corresponding index
        batch_frames = []
        batch_indices = []

        for frame_file, frame_index in tqdm(zip(frame_files, frame_indices), total=len(frame_files), desc=f'Fetching from {batch}'):
            frame_path = os.path.join(batch_dir, frame_file)
            frame = Image.open(frame_path)
            batch_frames.append(frame)
            batch_indices.append(frame_index)

        # Store frames and indices for this batch
        all_frames_by_batch.append(batch_frames)
        all_indices_by_batch.append(batch_indices)

        # Append to the overall lists
        all_frames.extend(batch_frames)
        all_indices.extend(batch_indices)

    logging.info(f"Fetched {len(all_frames)} frames and {len(all_indices)} corresponding indices.")

    # Return based on the 'batches' flag
    if batches:
        return all_frames_by_batch, all_indices_by_batch
    else:
        return [all_frames], [all_indices]

# Load Florence-2 model for vision-language tasks
# ! mkdir my_models/Florence_2 # Ceates folder/directory named 'my_models' and sub folder Florence_2 where florence2 models can be saved (e.g., Florence-2-large)
florence_models_dir = 'my_models/Florence_2'
model_id = 'microsoft/Florence-2-large'

# Load model and processor with GPU acceleration
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=florence_models_dir,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype='auto'
).eval().cuda()

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=florence_models_dir,
    trust_remote_code=True
)

def run_florence2_inference(image, task_prompt, text_input=None):
    """
    Runs multimodal inference on an image using the Florence-2 model.
    Supports both captioning and grounding depending on task prompt.

    Args:
        image (PIL.Image): The input image to run inference on.
        task_prompt (str): The task prompt that specifies the type of inference (e.g., "<CAPTION>" for captioning).
        text_input (str, optional): Additional text input to be appended to the task prompt (default is None).

    Returns:
        dict: A dictionary containing the results of the inference, where keys correspond to
              task prompts and values are the corresponding model outputs.
    """
    prompt = task_prompt if text_input is None else task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def clean_text(text):
    """
    Cleans and tokenizes input text by removing stopwords, punctuation, and converting to lowercase.
    Args:
        text (str): The input text to clean and process. It should be a string containing the text
                    that needs to be tokenized and cleaned.
    Returns:
        set: A set of filtered and tokenized words from the input text. The words are converted to lowercase
             and stopwords are removed.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return set(filtered_words)

def compare_texts(user_text_input, caption_text_input):
    """
    Compares the user's input text with the caption text by cleaning and tokenizing both inputs.
    Calculates the match percentage between the two texts based on the common words.

    Args:
        user_text_input (str): The user's input text to be compared with the caption. It is typically a
                                prompt or query that needs to be matched with the caption.
        caption_text_input (str): The caption text generated or obtained from the image inference.
                                  This text will be compared against the `user_text_input`.

    Returns:
        float: The percentage of matched words between the `user_text_input` and `caption_text_input`.
               Returns 0 if no matches are found.
    """
    # Clean and tokenize both inputs
    user_words = clean_text(user_text_input) 
    caption_words = clean_text(caption_text_input)

    # Find number of matched words
    matched_words = user_words.intersection(caption_words) # Find matching/common words
    matched_words = list(matched_words)
    match_count = len(matched_words) # Total no.of matching/common words
    total_user_words = len(user_words) # Total user tokens/words
    total_caption_words = len(caption_words) # Total caption tokens/words

    # Calculate match percentage
    if total_user_words > 0 and total_user_words > total_caption_words:
      match_percent = (match_count / total_caption_words) * 100 
    elif total_caption_words > 0 and total_caption_words > total_user_words:
      match_percent = (match_count / total_user_words) * 100
    else:
      match_percent = 0

    # Print result if there’s at least one match
    # if match_count > 0:
    #   if total_user_words < total_caption_words:
    #     print(f"Matched {match_count} user words out of {total_user_words} caption words ({match_percent:.2f}%)")
    #   elif total_user_words > total_caption_words:
    #     print(f"Matched {match_count} caption words out of {total_caption_words} user words ({match_percent:.2f}%)")
    #   # print(f"caption text input: {caption_text_input}")
    #   # print(f"user prompt: {user_text_input}")
    #   print(f"Caption tokens: {sorted(caption_words)}")
    #   print(f"User tokens: {sorted(user_words)}")
    #   print(f"Matched words: {sorted(matched_words)}")
      
    # else:
    #   print('Zero matched words')

    return match_percent

def plot_bbox(image, data):
    """
    Plots bounding boxes on a given image with label annotations. This function is useful for grounding tasks
    where object detection results are visualized on the image.

    Args:
        image (PIL.Image): The image on which the bounding boxes will be plotted. It should be a PIL Image object.
        data (dict): A dictionary containing bounding box and label information.
                     The dictionary should have the following keys:
                     - 'bboxes' (list of tuples): List of bounding boxes, where each box is defined by a
                                                   tuple (x1, y1, x2, y2) representing the top-left and
                                                   bottom-right corners of the box.
                     - 'labels' (list of str): List of labels corresponding to each bounding box in 'bboxes'.
    Returns:
        None: The function modifies the `image` object in place by drawing bounding boxes and labels. It does not
              return any value.
    """
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    plt.show()

# Define a colormap for drawing annotations
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws polygons on an image based on segmentation output. Optionally, regions can be filled with color.

    Args:
        image (PIL.Image): The image on which the polygons will be drawn. It should be a PIL Image object.
        prediction (dict): A dictionary containing segmentation output, which should include:
                           - 'polygons' (list of list of tuples): List of polygons, where each polygon is represented
                             as a list of (x, y) coordinates.
                           - 'labels' (list of str): List of labels corresponding to each polygon in 'polygons'.
        fill_mask (bool, optional): A flag that determines whether the polygon regions should be filled with color.
                                    If `True`, each polygon will be filled with a random color; otherwise,
                                    only the outline will be drawn. Default is `False`.
    Returns:
        None: The function modifies the `image` object in place by drawing polygons. It does not return any value.
    """
    draw = ImageDraw.Draw(image)
    scale = 1

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    display(image)

# def find_object_segments_v3(video_path, frames_batches, frame_indices_batches, user_text_input,
#                          detail_level='high', thresholds=np.array([85, 90, 95], dtype=np.float32),
#                          plot_matching_frames=False, batch_size=5):
def find_object_segments(video_path, frames_batches, frame_indices_batches, user_text_input,
                         detail_level='high', thresholds=np.array([85, 90, 95], dtype=np.float32),
                         plot_matching_frames=False):  
    """
    Finds the start and end timestamps of segments in the video that match the user's input text/prompt based on
    image captions. The function performs inference on each frame and compares the caption text with the
    user's input to identify matching segments.

    Args:
        video_path (str): The path to the input video file. This is used to extract timestamps for the segments.
        frames_batches (list of lists of PIL.Images): A list of lists video frames to process, extracted from the video.
        frame_indices_batches (list of lists of ints): A list of lists of indices corresponding to the video frames.
        user_text_input (str): The user's input text or prompt to match against the captions of each frame.
        detail_level (str, optional): Level of detail for inference ('high', 'medium', 'low'). Default is 'high'.
        thresholds (np.ndarray, optional): Match percentage thresholds in the order:
                                           [<CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>]. Default is [85., 90., 95.].
        plot_matching_frames (bool, optional): If True, displays bounding boxes on the start and end matching frames.
    Returns:
        list of dict: List of dictionaries with 'start', 'end' timestamps.
    """
    thresholds = sorted(thresholds)
    thresholds_dict = {
        'high': thresholds[-1].item(),
        'medium': thresholds[1].item(),
        'low': thresholds[0].item()
    }

    task_prompts = {
        'high': '<MORE_DETAILED_CAPTION>',
        'medium': '<DETAILED_CAPTION>',
        'low': '<CAPTION>'
    }

    segments = []
    segment_visuals = []
    match_started = False
    current_detail_level = detail_level
    batch_num = 0

    progress_bar = tqdm(total=len(frames_batches), unit=" batches", desc=f'Running inference on {len(frames_batches)} batches of frames')

    for batch_frames_list, batch_indices_list in zip(frames_batches, frame_indices_batches):
        try:
            for frame, index in tqdm(zip(batch_frames_list, batch_indices_list), desc=f' Batch no. {batch_num}', total=len(batch_frames_list)):
                task_prompt = task_prompts[current_detail_level]

                start_time = time.time()
                results = run_florence2_inference(frame, task_prompt)
                inference_time = time.time() - start_time

                if current_detail_level == 'high' and inference_time > 1:
                    print(f" Inference is slow with high detail level ({inference_time:.2f}s/frame), switching to medium...")
                    current_detail_level = 'medium'
                elif current_detail_level == 'medium' and inference_time > 1:
                    print(f" Still slow with medium detail level ({inference_time:.2f}s/frame), switching to low...")
                    current_detail_level = 'low'

                caption_text_input = results[task_prompt]
                match_percent = compare_texts(user_text_input, caption_text_input)

                if match_percent is not None and match_percent >= thresholds_dict[current_detail_level]:
                    if not match_started:
                        start_index = index
                        start_frame = frame
                        match_started = True
                        start_results = run_florence2_inference(start_frame, '<CAPTION_TO_PHRASE_GROUNDING>', user_text_input)

                    end_index = index
                    end_frame = frame
                    end_results = run_florence2_inference(end_frame, '<CAPTION_TO_PHRASE_GROUNDING>', user_text_input)

                elif match_started:
                    segments.append({
                        'start': get_timestamp_by_index(video_path, start_index),
                        'end': get_timestamp_by_index(video_path, end_index)
                    })

                    segment_visuals.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'start_results': start_results,
                        'end_results': end_results,
                        'start_index': start_index,
                        'end_index': end_index
                    })
                    print(f" First matching segment found at frame no. {start_index} in btach no. {batch_num}\n Start: {segments[-1]['start']}, End: {segments[-1]['end']} | Start frame index: {start_index} End frame index: {end_index}")
                    if plot_matching_frames:
                      print(f'Start frame no. {start_index}')
                      plot_bbox(start_frame, start_results['<CAPTION_TO_PHRASE_GROUNDING>'])
                      print(f'End frame no. {end_index}')
                      plot_bbox(end_frame, end_results['<CAPTION_TO_PHRASE_GROUNDING>'])
                      # for seg, vis in zip(segments, segment_visuals):
                      #     plot_start_end_bbox_side_by_side(
                      #         vis['start_frame'],
                      #         vis['end_frame'],
                      #         vis['start_index'],
                      #         vis['end_index'],
                      #         vis['start_results'],
                      #         vis['end_results']
                      #     )
                    match_started = False
                    user_input = input(" Do you want to continue finding more segments? (yes/no): ")
                    if user_input.lower() != 'yes':
                        progress_bar.update(len(batch_frames_list))
                        progress_bar.close()
                        print(f"Inference ended at frame no. {end_index}, batch {batch_num}")
                        # if plot_matching_frames:
                        #     for seg, vis in zip(segments, segment_visuals):
                        #         plot_start_end_bbox_side_by_side(
                        #             vis['start_frame'],
                        #             vis['end_frame'],
                        #             vis['start_index'],
                        #             vis['end_index'],
                        #             vis['start_results'],
                        #             vis['end_results']
                        #         )

                        return segments

        except (ValueError, AttributeError) as e:
            print(f" Error during inference: {e}")

        finally:
            progress_bar.update(len(batch_frames_list))
        batch_num += 1

    progress_bar.close()
    print(f"Inference ended at frame no. {end_index}, batch {batch_num}")
    if plot_matching_frames:
      print(f'Start frame no. {start_index}')
      plot_bbox(start_frame, start_results['<CAPTION_TO_PHRASE_GROUNDING>'])
      print(f'End frame no. {end_index}')
      plot_bbox(end_frame, end_results['<CAPTION_TO_PHRASE_GROUNDING>'])
        # for seg, vis in zip(segments, segment_visuals):
        #     plot_start_end_bbox_side_by_side(
        #         vis['start_frame'],
        #         vis['end_frame'],
        #         vis['start_index'],
        #         vis['end_index'],
        #         vis['start_results'],
        #         vis['end_results']
        #     )

    return segments


def get_timestamp_by_index(video_path, target_index):
    """
    Returns timestamp (in seconds) for a specific frame index.

    Args:
        video_path (str): Path to the video file.
        target_index (int): Frame index to fetch the timestamp for.

    Returns:
        float: Timestamp in seconds, or None if the index is invalid.
    """
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    logging.info(f"Extracting timestamp for frame {target_index}...")

    while cap.isOpened():
        frame_exists, _ = cap.read()
        if not frame_exists:
            break

        if frame_num == target_index:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.release()
            return round(timestamp_ms / 1000.0, 3)  # Convert to seconds with 3 decimals

        frame_num += 1

    cap.release()
    logging.warning(f"Frame index {target_index} not found in video.")
    return None

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

def edit_video(original_video_path, segments, output_video_path=None, fade_duration=0.5):
    """
    Extracts and concatenates video clips from relevant segments.

    Args:
        original_video_path (str): Path to original video.
        segments (list): List of {start, end} dicts.
        fade_duration (float): Duration of fadein/fadeout in seconds.
    """
    if output_video_path is None:
      output_video_path = edit_paths(original_video_path)  # Create output path using original video/audio file path if it is None
    else:
      output_video_path = edit_paths(output_video_path)  # Create a new unique output path if it already exists, using the same base.

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

def main():
    """
    Main pipeline to process video, extract relevant segments based on prompt,
    and create highlight clips.
    """
    video_path = 'your_video.mp4'  # Path to the input video file
    video_frames_batches_dir = '/video_frames'  # Directory where folder of frames will be saved in batches
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract the video name without the extension
    output_video_path = None

    sample_interval = adjust_sample_interval(video_path)  # Dynamically adjust the sample interval based on video length
    batch_size = determine_chunk_size()  # Dynamically adjust the batch size based on available system memory
    video_info = sv.VideoInfo.from_video_path(video_path)  # Extract video information (such as total frames)

    # Step 1: Save frames in batches with the specified interval and batch size
    save_frames_and_indices_in_batches(
        video_path=video_path,
        total_frames=video_info.total_frames,  # Total frames in the video
        batch_size=batch_size,  # Batch size for saving frames
        base_dir=video_frames_batches_dir,  # Directory where batches will be saved
        sample_interval=sample_interval  # Interval at which frames will be sampled (e.g., every 10th frame)
    )

    # Step 2: Load saved frames and their indices from the batch directories
    frames_and_indices_batches = fetch_frames_and_indices_from_batches(video_frames_batches_dir, video_path, True)
    frames_batches, frame_indices_batches = frames_and_indices_batches[0], frames_and_indices_batches[1]

    # Step 3: User input text for finding relevant segments (e.g., a description)
    user_text_input = 'your prompt'

    # Step 4: Find video segments that match the user's prompt input by comparing it with captions generated from frames
    matching_segments = find_object_segments(video_path, frames_batches, frame_indices_batches, user_text_input)

    # Step 5: Edit the video by extracting the matching segments and creating a highlight video
    edit_video(video_path, matching_segments)  # Edit the video with the found segments and save the output

# Entry point
if __name__ == '__main__':
    main()
