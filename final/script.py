import openai

openai.api_key = 'your_openai_api_key_here'

def generate_stream_script(prompt, max_tokens=500):
    """
    Generates a stream script based on the given prompt.
    Args:
        prompt (str): Prompt for script generation.
        max_tokens (int): Maximum tokens for the generated script.
    Returns:
        str: Generated script.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or use "gpt-4"
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response['choices'][0]['text'].strip()

