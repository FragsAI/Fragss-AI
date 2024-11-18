import requests

# API key for Elevenlabs
api_key = "sk_516b6416f820c0c196426bb27728e2ff31b95db18955b1c1"
url = "https://api.elevenlabs.io/v1/text-to-speech"

def generate_voiceover(text, voice="Bella", output_path="voiceover.mp3"):
    """
    Generates a voiceover for the given text using Elevenlabs API.
    Args:
        text (str): Text for the voiceover.
        voice (str): Voice name.
        output_path (str): Path to save the voiceover file.
    Returns:
        str: Status of the generation.
    """
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "voice": voice,
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return f"Voiceover generated and saved as '{output_path}'."
    else:
        return f"Error: {response.status_code}, {response.text}"
