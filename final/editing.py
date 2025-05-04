import requests

# Shotstack API keys
sandbox_key = "uxkEreSYgnrPj4q4hxVuJYqA0jwDoQ3SjIvwiEEV"
production_key = "xB9YvCM80apOAjKyfAJCwHKYsc0XuQq4dKpDv3Th"

def edit_video(clip_url, soundtrack_url=None, title="Gaming Stream Clip", resolution="hd", length=10):
    """
    Edits a video using the Shotstack API with user-provided inputs.
    Args:
        clip_url (str): URL of the video clip.
        soundtrack_url (str): URL of the soundtrack.
        title (str): Title for the clip.
        resolution (str): Video resolution (e.g., 'hd').
        length (int): Length of the clip in seconds.
    Returns:
        dict or str: Response from the Shotstack API or error details.
    """
    headers = {
        "x-api-key": sandbox_key,  # Use production_key for production
        "Content-Type": "application/json"
    }

    data = {
        "timeline": {
            "soundtrack": {
                "src": soundtrack_url or "https://shotstack-assets.s3-ap-southeast-2.amazonaws.com/music/free/music.mp3",
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
                            "length": length,
                            "title": title
                        }
                    ]
                }
            ]
        },
        "output": {
            "format": "mp4",
            "resolution": resolution
        }
    }
    
    response = requests.post("https://api.shotstack.io/stage/render", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Contains render ID and status
    else:
        return f"Error: {response.status_code}, {response.text}"
