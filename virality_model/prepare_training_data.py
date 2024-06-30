import numpy as np
import pandas as pd
import re
from textblob import TextBlob

# Load data from CSV
df = pd.read_csv('/Users/kesinishivaram/FragsAI/minecraft_shorts.csv')

# Function to convert ISO 8601 duration to seconds
def iso8601_to_seconds(duration):
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes) if minutes else 0
    seconds = int(seconds) if seconds else 0
    return hours * 3600 + minutes * 60 + seconds

# Preprocess data
df['duration'] = df['duration'].apply(iso8601_to_seconds)

# Function to perform sentiment analysis on top comment
def get_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0

# Apply sentiment analysis
df['sentiment'] = df['top_comment'].apply(get_sentiment)

# Extract features
features = df[['duration', 'views', 'likes', 'comments', 'sentiment']]

# Define virality score
df['virality_score'] = (df['views'] + 2 * df['likes'] + 3 * df['comments']) / df['duration']
labels = df['virality_score']

# Save features and labels as numpy arrays
np.save('video_features.npy', features.values)
np.save('virality_labels.npy', labels.values)

x = np.load('video_features.npy')
x = np.load('virality_labels.npy')
print(x)

print("Features and labels have been saved as 'video_features.npy' and 'virality_labels.npy'.")
