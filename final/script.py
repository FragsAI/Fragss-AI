import openai

# GPT API key is imported directly
openai.api_key = 'your_openai_api_key_here'

def generate_stream_script(prompt):
    response = openai.Completion.create(
      engine="text-davinci-003",  # Or use "gpt-4"
      prompt=prompt,
      max_tokens=500
    )
    script = response['choices'][0]['text'].strip()
    return script

# Example usage
script = generate_stream_script("Generate a funny script for a gaming stream highlight")
print(script)
