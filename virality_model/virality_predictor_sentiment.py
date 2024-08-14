import ssl
import os
import re
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure the VADER lexicon is downloaded
#nltk.download('vader_lexicon')

def extract_text_from_srt(srt_content: str) -> str:
    cleaned_text = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', srt_content)
    cleaned_text = re.sub(r'\n\d+\n', '\n', cleaned_text)
    cleaned_text = cleaned_text.replace('\n', ' ').strip()
    return cleaned_text

def sentiment_analysis(text: str) -> dict:
    vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer.polarity_scores(text)

def load_and_concatenate_csvs(folder_path: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['subtitle_sentiment'] = df['subtitles'].apply(lambda x: sentiment_analysis(x) if pd.notnull(x) else {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
    df['top_comment_sentiment'] = df['top_comment'].apply(lambda x: sentiment_analysis(x) if pd.notnull(x) else {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})

    df = pd.concat([df.drop(['subtitle_sentiment', 'top_comment_sentiment'], axis=1),
                    df['subtitle_sentiment'].apply(pd.Series),
                    df['top_comment_sentiment'].apply(pd.Series)], axis=1)

    df['duration_seconds'] = df['duration'].apply(lambda x: int(re.search(r'\d+S', x).group(0).replace('S', '')) if 'S' in x else 0)
    
    df['virality_score'] = (df['views'] - df['views'].min()) / (df['views'].max() - df['views'].min()) * 100
    
    return df

def prepare_data(df: pd.DataFrame, target_col: str):
    features = df.drop(columns=[target_col, 'id', 'subtitles', 'top_comment', 'duration'])
    target = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    return model

def preprocess_transcript(srt_file_path: str, model) -> pd.DataFrame:
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()
    
    transcript = extract_text_from_srt(srt_content)
    
    sentiment_scores = sentiment_analysis(transcript)
    
    # Add placeholder values for the missing features
    sentiment_scores['comments'] = 0
    sentiment_scores['duration_seconds'] = 0
    sentiment_scores['likes'] = 0
    sentiment_scores['views'] = 0
    
    # Get feature names from the model to ensure the correct order
    feature_names = model.feature_names_in_
    sentiment_df = pd.DataFrame([[sentiment_scores[feature] for feature in feature_names]], columns=feature_names)
    
    return sentiment_df

def predict_virality(srt_file_path: str, model_filename: str = 'virality_model.pkl'):
    model = joblib.load(model_filename)
    features = preprocess_transcript(srt_file_path, model)
    
    virality_score = model.predict(features)
    
    return virality_score[0]

if __name__ == "__main__":
    # TRAINING AND SAVING THE MODEL
    folder_path = "/Users/kesinishivaram/FragsAI/youtube_data"
    
    df = load_and_concatenate_csvs(folder_path)
    
    df = preprocess_data(df)
    
    target_col = 'virality_score'
    
    X_train, X_test, y_train, y_test = prepare_data(df, target_col)
    
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    model_filename = 'virality_model.pkl'
    joblib.dump(model, model_filename)
    
    # PREDICTING VIRALITY OF NEW CLIPS BASED ON TRANSCRIPT FILE
    srt_file_path = '/Users/kesinishivaram/FragsAI/clips/sub-clips/clip_10_aspect_ratio.mp4.en.srt'  # Replace with your .srt file path
    virality_score = predict_virality(srt_file_path) * 10000
    print(f'Predicted Virality Score: {virality_score}')
