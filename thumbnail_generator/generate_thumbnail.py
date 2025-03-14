import cv2
import numpy as np
import ast
from openai import OpenAI
import json
from moviepy.editor import *

from virality_model.predict_virality import extract_frames, predict_actions

client = OpenAI(api_key="")

def generate_thumbnail_background_from_selected_timestap(video_path, output_path, time_sec=7, thumbnail_size=(1280, 720)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    success, frame = cap.read()
    
    if success:
        thumbnail_background = cv2.resize(frame, thumbnail_size)
        thumbnail_background_path = video_path.replace('.mp4', '_background.jpg')  
        cv2.imwrite(thumbnail_background_path, thumbnail_background)
        print(f"Thumbnail background generated at: {thumbnail_background_path}")
        return thumbnail_background_path    

    else:
        print("Error generating thumbnail background")
        return None
    
def select_timestamp_best_frame(clip_path):
    frames = np.array(extract_frames(clip_path))
    predictions = predict_actions(frames)

    # Verify predictions shape explicitly:
    predictions = np.array(predictions)
    print(f"Predictions shape: {predictions.shape}")
    frame_confidences = np.max(predictions, axis=1)
    best_frame_index = np.argmax(frame_confidences)

    return clip_path, best_frame_index

def generate_thumbnail_options(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or use "gpt-4"
        messages=[
            {"role": "system", "content": "Generate structured JSON with keys 'text_options' and 'icon_options' for a video thumbnail. "
                                        "'text_options' should include cv2 putText params: 'text', 'font_scale', 'font_thickness', 'text_color' (rgb string format), 'shadow_color' (rgb string format), 'position' (text position x and y). "
                                        "'icon_options' should include cv2 putText tions params: 'icon_type', 'size', 'color (rgb string format)', 'position' (icon center position x and y)."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    content = response.choices[0].message.content
    try:
        options = json.loads(content)
    except json.JSONDecodeError:
        print("GPT response was not valid JSON:", content)
        return None, None
    
    text_options = options.get('text_options')
    print("Text options:", text_options)
    icon_options = options.get('icon_options')
    print("Icon options:", icon_options)
    return text_options, icon_options


def add_text_and_icon(image_path, text_options=None, icon_options=None):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error loading image")
        return None
    
    if text_options:
        text = text_options.get("text", "Default Title")
        font = text_options.get("font", cv2.FONT_HERSHEY_SIMPLEX)
        font_scale = text_options.get("font_scale", 2)
        font_thickness = text_options.get("font_thickness", 5)
        text_color = ast.literal_eval(text_options['text_color']) if text_options['text_color'] else (0, 0, 0)
        shadow_color = ast.literal_eval(text_options['shadow_color']) if text_options['shadow_color'] else (0, 0, 0)
        text_x = text_options["position"]['x']
        text_y = text_options["position"]['y']

        cv2.putText(image, text, (text_x + 2, text_y + 2), font, font_scale, shadow_color, font_thickness+2, cv2.LINE_AA)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if icon_options:
        icon_type = icon_options.get("icon_type", "play") 
        icon_size = int(icon_options.get("size", 100))
        center_x = icon_options["position"]['x']
        center_y = icon_options["position"]['y']
        icon_color = ast.literal_eval(icon_options['color']) if icon_options['color'] else (0, 0, 0)

        if icon_type == "play": 
            pts = np.array([
                (center_x - icon_size // 2, center_y - icon_size // 2),
                (center_x - icon_size // 2, center_y + icon_size // 2),
                (center_x + icon_size // 2, center_y)
            ], np.int32)
            cv2.fillPoly(image, [cv2.convexHull(pts)], icon_color)

        elif icon_type == "circle":  
            cv2.circle(image, (center_x, center_y), icon_size // 2, icon_color, -1)

        elif icon_type == "square":
            top_left = (center_x - icon_size // 2, center_y - icon_size // 2)
            bottom_right = (center_x + icon_size // 2, center_y + icon_size // 2)
            cv2.rectangle(image, top_left, bottom_right, icon_color, -1)

    return cv2.imwrite(image_path.replace('_background', ''), image)

if __name__ == '__main__':
    video_path = "uploads/videoplayback (online-video-cutter.com).mp4"
    output_path = "thumbnail_custom.jpg"

    text_options = {
        "text": "My Custom Video",
        "font": cv2.FONT_HERSHEY_DUPLEX,
        "font_scale": 2.5,
        "font_thickness": 6,
        "text_color": (255, 255, 0),
        "shadow_color": (0, 0, 0),
        "position": (100, 150) 
    }

    icon_options = {
        "icon_type": "play", 
        "size": 150,
        "position": (640, 360),
        "color": (0, 255, 0)
    }

    generate_thumbnail_background_from_selected_timestap(video_path, output_path, time_sec=10, text_options=text_options, icon_options=icon_options)