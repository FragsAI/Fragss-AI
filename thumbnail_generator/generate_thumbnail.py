import cv2
import numpy as np
import os

def generate_video_thumbnail(video_path, output_path, time_sec=0, thumbnail_size=(1280, 720)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    success, frame = cap.read()
    
    if success:
        thumbnail_background = cv2.resize(frame, thumbnail_size)
        thumbnail_background_path = os.path.join(output_path, "thumbnail_background.jpg")
        cv2.imwrite(thumbnail_background_path, thumbnail_background)
        print(f"Thumbnail generated at: {output_path}")
        return thumbnail_background_path    

    else:
        print("Error generating thumbnail")
        return None

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
        text_color = text_options.get("text_color", (255, 255, 255))
        shadow_color = text_options.get("shadow_color", (0, 0, 0))
        text_x, text_y = text_options.get("position", (50, 100))

        cv2.putText(image, text, (text_x+2, text_y+2), font, font_scale, shadow_color, font_thickness+2, cv2.LINE_AA)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if icon_options:
        icon_type = icon_options.get("icon_type", "play") 
        icon_size = icon_options.get("size", 100)
        center_x, center_y = icon_options.get("position", (image.shape[1] // 2, image.shape[0] // 2))
        icon_color = icon_options.get("color", (255, 255, 255))

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

    generate_video_thumbnail(video_path, output_path, time_sec=10, text_options=text_options, icon_options=icon_options)