from openai import OpenAI

client = OpenAI()

def generate_stream_script(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # or your preferred model
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text
# Example usage
script = generate_stream_script("Generate a funny script for a gaming stream highlight")
print(script)
