import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

# Download VADER lexicon (if not already downloaded)
#nltk.download('vader_lexicon')

def extract_text_from_srt(srt_content: str) -> str:
    # Remove timestamps and line numbers using regular expressions
    cleaned_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', srt_content)
    # Remove any remaining line numbers
    cleaned_text = re.sub(r'\n\d+\n', '\n', cleaned_text)
    # Remove extra newlines
    cleaned_text = cleaned_text.replace('\n', ' ').strip()
    return cleaned_text

def analyze_srt_sentiments_in_folder(folder_path: str):
    # Initialize the VADER sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # Dictionary to hold the sentiment results for each file
    sentiment_results = {}

    # Loop through each file in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".srt"):  # Ensure only .srt files are processed
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                srt_content = file.read()
                
                # Extract text content from the .srt file
                transcript = extract_text_from_srt(srt_content)
                
                # Perform sentiment analysis on the extracted transcript
                sentiment_scores = vader_analyzer.polarity_scores(transcript)
                
                # Store the result in the dictionary
                sentiment_results[filename] = sentiment_scores
                
                # Print the results for each file
                print(f"Sentiment analysis for {filename}:")
                for k, v in sentiment_scores.items():
                    print(f"  {k}: {v}")
                print()  # Add a newline for better readability

    return sentiment_results

# Example usage
if __name__ == "__main__":
    folder_path = "/Users/kesinishivaram/FragsAI/clips/sub-clips"
    results = analyze_srt_sentiments_in_folder(folder_path)
    