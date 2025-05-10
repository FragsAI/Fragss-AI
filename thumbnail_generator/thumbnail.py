import cv2
import numpy as np
import ast
import json
import requests
from openai import OpenAI
from moviepy.editor import VideoFileClip
from virality_model.predict_virality import extract_frames, predict_actions

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_API_KEY")

# -------- Mode 1: Best Frame from Clip -------- #
def select_best_frame(video_path):
    frames = np.array(extract_frames(video_path))
    predictions = predict_actions(frames)
    frame_confidences = np.max(np.array(predictions), axis=1)
    best_index = np.argmax(frame_confidences)
    return video_path, best_index

def generate_thumbnail_background(video_path, output_path, time_sec=7, size=(1280, 720)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    success, frame = cap.read()
    cap.release()
    if success:
        thumbnail = cv2.resize(frame, size)
        path = output_path.replace('.jpg', '_background.jpg')
        cv2.imwrite(path, thumbnail)
        return path
    return None

# -------- Mode 2: Prompt-Based DALL·E 3 -------- #
def generate_dalle3_thumbnail(prompt, output_path="thumbnail_dalle3.jpg"):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1280x720",
        quality="standard",
        n=1
    )
    image_url = response.data[0].url
    image_data = requests.get(image_url).content
    with open(output_path, "wb") as f:
        f.write(image_data)
    return output_path

# -------- Mode 3: Sketch + Prompt -------- #
def generate_from_sketch(sketch_path, prompt, output_path="thumbnail_from_sketch.png"):
    response = client.images.edit(
        image=open(sketch_path, "rb"),
        prompt=prompt,
        size="1280x720",
        n=1
    )
    image_url = response.data[0].url
    image_data = requests.get(image_url).content
    with open(output_path, "wb") as f:
        f.write(image_data)
    return output_path

# -------- Text + Icon Overlays -------- #
def add_text_and_icon(image_path, text_options=None, icon_options=None):
    image = cv2.imread(image_path)
    if image is None:
        return None

    if text_options:
        text = text_options.get("text", "Viral Gaming Moment")
        font = text_options.get("font", cv2.FONT_HERSHEY_SIMPLEX)
        scale = text_options.get("font_scale", 2)
        thickness = text_options.get("font_thickness", 5)
        color = ast.literal_eval(text_options['text_color']) if text_options.get('text_color') else (255, 255, 255)
        shadow = ast.literal_eval(text_options['shadow_color']) if text_options.get('shadow_color') else (0, 0, 0)
        pos = text_options["position"]
        x, y = pos['x'], pos['y']
        cv2.putText(image, text, (x + 2, y + 2), font, scale, shadow, thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    if icon_options:
        icon = icon_options.get("icon_type", "play")
        size = int(icon_options.get("size", 100))
        pos = icon_options["position"]
        cx, cy = pos['x'], pos['y']
        color = ast.literal_eval(icon_options['color']) if icon_options.get('color') else (0, 255, 0)

        if icon == "play":
            pts = np.array([
                (cx - size // 2, cy - size // 2),
                (cx - size // 2, cy + size // 2),
                (cx + size // 2, cy)
            ], np.int32)
            cv2.fillPoly(image, [cv2.convexHull(pts)], color)
        elif icon == "circle":
            cv2.circle(image, (cx, cy), size // 2, color, -1)
        elif icon == "square":
            cv2.rectangle(image, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), color, -1)

    output_final = image_path.replace("_background", "")
    cv2.imwrite(output_final, image)
    return output_final

# -------- GPT Assist (Text/Icon Generator) -------- #
def generate_thumbnail_overlays(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Return JSON with keys 'text_options' and 'icon_options'. Each should include overlay details for video thumbnail as described."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    content = response.choices[0].message.content
    try:
        options = json.loads(content)
        return options.get("text_options"), options.get("icon_options")
    except json.JSONDecodeError:
        print("Invalid JSON from GPT:", content)
        return None, None
