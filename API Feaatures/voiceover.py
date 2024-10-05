import requests

# API key for Elevenlabs
api_key = "sk_516b6416f820c0c196426bb27728e2ff31b95db18955b1c1"
url = "https://api.elevenlabs.io/v1/text-to-speech"

def generate_voiceover(text, voice="Bella"):
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "voice": voice,  # Change voice based on preferences
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open("voiceover.mp3", "wb") as f:
            f.write(response.content)
        return "Voiceover generated and saved as 'voiceover.mp3'."
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
generate_voiceover("Welcome to the gaming stream highlight!")
