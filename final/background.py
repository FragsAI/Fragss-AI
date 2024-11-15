import requests

# Stable Diffusion API key
api_key = "GE9sBm4fepU46hfVhLAP4AoNvwdqb4ulXp4y6ubV8IkhX1CgTKsxjhGOCU3t"
url = "https://api.stablediffusionapi.com/v1/generate"

def generate_background(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "steps": 30,  # Number of steps for better quality
        "cfg_scale": 7.5,
        "width": 1024,
        "height": 576,
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open("background_image.png", "wb") as f:
            f.write(response.content)
        return "Background image saved as 'background_image.png'."
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
generate_background("A futuristic gaming setup with neon lights")
