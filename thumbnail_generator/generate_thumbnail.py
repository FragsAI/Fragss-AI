import cv2
import os

def generate_video_thumbnail(video_path, output_path, time_sec=0, thumbnail_size=(1280, 720)):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    success, frame = cap.read()
    
    if success:
        thumbnail = cv2.resize(frame, thumbnail_size)
        cv2.imwrite(output_path, thumbnail)
        print(f"Thumbnail generated at: {output_path}")
        return output_path    

    else:
        print("Error generating thumbnail")
        return None
    
if __name__ == '__main__':
    video_path = "uploads/videoplayback (online-video-cutter.com).mp4"
    output_path = "thumbnail.jpg"
    generate_video_thumbnail(video_path, output_path, time_sec=10)  # Generate thumbnail at 10 seconds