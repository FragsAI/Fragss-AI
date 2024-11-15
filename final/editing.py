import requests

# Shotstack API keys
sandbox_key = "uxkEreSYgnrPj4q4hxVuJYqA0jwDoQ3SjIvwiEEV"
production_key = "xB9YvCM80apOAjKyfAJCwHKYsc0XuQq4dKpDv3Th"

def edit_video(clip_url, title="Gaming Stream Clip"):
    headers = {
        "x-api-key": sandbox_key,  # Use production_key for production
        "Content-Type": "application/json"
    }

    data = {
        "timeline": {
            "soundtrack": {
                "src": "https://shotstack-assets.s3-ap-southeast-2.amazonaws.com/music/free/music.mp3",  # Example audio
            },
            "tracks": [
                {
                    "clips": [
                        {
                            "asset": {
                                "type": "video",
                                "src": clip_url
                            },
                            "start": 0,
                            "length": 10,  # Edit based on your clip
                            "title": title
                        }
                    ]
                }
            ]
        },
        "output": {
            "format": "mp4",
            "resolution": "hd"
        }
    }
    
    response = requests.post("https://api.shotstack.io/stage/render", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Contains render ID and status
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
edit_video("https://your-video-url.com/clip.mp4")
