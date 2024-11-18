import moviepy.editor as mp
import logging
import os

def enhance_video_aspect_ratio(input_video, output_folder, desired_aspect_ratio=9/16, target_width=720, target_height=1280):
    """
    Enhances the video by adjusting its aspect ratio for platforms like TikTok/Shorts.
    Args:
        input_video (str): Path to the input video.
        output_folder (str): Folder to save the adjusted video.
        desired_aspect_ratio (float): Desired aspect ratio (e.g., 9/16 for TikTok).
        target_width (int): Target width for the output video.
        target_height (int): Target height for the output video.
    Returns:
        str: Path to the output video.
    """
    try:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Load the video
        video = mp.VideoFileClip(input_video)
        
        # Get current dimensions
        width, height = video.size
        current_aspect_ratio = width / height

        # Adjust dimensions based on the desired aspect ratio
        if current_aspect_ratio > desired_aspect_ratio:
            # Width is too large, resize by width and pad top and bottom
            new_width = target_width
            new_height = int(new_width / desired_aspect_ratio)
            video = video.resize(width=new_width)
            padding_top = (new_height - height) // 2
            video = video.margin(top=padding_top, bottom=padding_top, color=(0, 0, 0))  # Black padding
        else:
            # Height is too large, resize by height and pad left and right
            new_height = target_height
            new_width = int(new_height * desired_aspect_ratio)
            video = video.resize(height=new_height)
            padding_left = (new_width - width) // 2
            video = video.margin(left=padding_left, right=padding_left, color=(0, 0, 0))  # Black padding

        # Define output video path
        output_video = os.path.join(output_folder, os.path.basename(input_video).replace(".mp4", "_aspect_ratio.mp4"))
        
        # Write the video file with audio
        video.write_videofile(output_video, codec="libx264", audio_codec="aac")

        logging.info(f"Video enhanced for aspect ratio {desired_aspect_ratio}: {output_video}")
        return output_video
    except Exception as e:
        logging.error(f"Error enhancing video aspect ratio: {e}")
        return None
