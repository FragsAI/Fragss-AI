import requests

# Stable Diffusion API key
api_key = "GE9sBm4fepU46hfVhLAP4AoNvwdqb4ulXp4y6ubV8IkhX1CgTKsxjhGOCU3t"
url = "https://api.stablediffusionapi.com/v1/generate"

def generate_background(prompt, output_path="background_image.png", width=1024, height=576):
    """
    Generates a background image based on a user-provided prompt.
    Args:
        prompt (str): Description of the background.
        output_path (str): Path to save the generated image.
        width (int): Width of the image.
        height (int): Height of the image.
    Returns:
        str: Status of the generation.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "steps": 30,
        "cfg_scale": 7.5,
        "width": width,
        "height": height,
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        return f"Background image saved as '{output_path}'."
    else:
        return f"Error: {response.status_code}, {response.text}"
