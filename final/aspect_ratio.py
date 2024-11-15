import moviepy.editor as mp
import logging
import os

def enhance_video_for_tiktok(input_video):
    try:
        # Load the video
        video = mp.VideoFileClip(input_video)
        
        # Define the desired aspect ratio for TikTok/Shorts (9:16)
        desired_aspect_ratio = 9 / 16
        
        # Get current dimensions
        width, height = video.size
        current_aspect_ratio = width / height
        
        if current_aspect_ratio > desired_aspect_ratio:
            # Width is too large, resize by width and pad top and bottom
            new_width = 720
            new_height = int(new_width / desired_aspect_ratio)
            video = video.resize(width=new_width)
            video = video.margin(top=(new_height - height) // 2, bottom=(new_height - height) // 2)
        else:
            # Height is too large, resize by height and pad left and right
            new_height = 1280
            new_width = int(new_height * desired_aspect_ratio)
            video = video.resize(height=new_height)
            video = video.margin(left=(new_width - width) // 2, right=(new_width - width) // 2)

        # Define output video path
        output_video = os.path.splitext(input_video)[0] + "_aspect_ratio.mp4"
        
        # Write the video file with audio
        video.write_videofile(output_video)

        logging.info(f"Video enhanced for TikTok/Shorts: {output_video}")
        return output_video
    except Exception as e:
        logging.error(f"Error enhancing video: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_video_path = "/Users/kesinishivaram/FragsAI/clips/clip_2.mp4"
    enhance_video_for_tiktok(input_video_path)
