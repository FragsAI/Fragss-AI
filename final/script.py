from openai import OpenAI

client = OpenAI(api_key="")

def generate_script(prompt):
    response = client.chat.completions.create(
        model="gpt-4",  # or your preferred model
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_stream_script(prompt):
    title_prompt = f"Generate a title for the topic: {prompt}"
    title = generate_script(title_prompt)
    
    script_prompt = f"Generate a script for my youtube video based on the title: {title} and on the topic: {prompt}"
    script = generate_script(script_prompt)
    return f"Title: {title}\n\nScript:\n{script}"

# Example usage
# script = generate_stream_script("CS 2 gaming stream highlight")  # Example, change to your preference
# print("Script: ", script)
