import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.utils import shuffle

# Load the dataset
data = pd.read_csv('dataset.csv') # Once we have the dataset

# Handle missing values
data = data.dropna()

# Random data till we have the dataset based on actions

# Convert dates to datetime
data['upload_date'] = pd.to_datetime(data['upload_date'])

# Placeholder for sentiment analysis
data['title_sentiment'] = np.random.rand(len(data))
data['description_sentiment'] = np.random.rand(len(data))

# Extract TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=100)
title_tfidf = tfidf_vectorizer.fit_transform(data['title']).toarray()
description_tfidf = tfidf_vectorizer.fit_transform(data['description']).toarray()

# Combine features
features = pd.concat([
    data[['views', 'duration', 'title_sentiment', 'description_sentiment']],
    pd.DataFrame(title_tfidf, index=data.index),
    pd.DataFrame(description_tfidf, index=data.index)
], axis=1)

# Target variable
target = data['views']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a regression model
regressor = xgb.XGBRegressor(objective='reg:squarederror')
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Regression Model Mean Squared Error: {mse}")

# Preparing data for ranking
data['pair_id'] = data.index

# Creating pairwise ranking data
pairs = []
for i in range(len(data) - 1):
    for j in range(i + 1, len(data)):
        pairs.append((data.iloc[i], data.iloc[j]))

# Creating the training set for ranking
rank_features = []
rank_labels = []

for a, b in pairs:
    if a['views'] > b['views']:
        rank_features.append(a.drop('views').values - b.drop('views').values)
        rank_labels.append(1)
    else:
        rank_features.append(b.drop('views').values - a.drop('views').values)
        rank_labels.append(-1)

rank_features = np.array(rank_features)
rank_labels = np.array(rank_labels)

# Train a ranking SVM
rank_svm = SVC(kernel='linear')
rank_svm.fit(rank_features, rank_labels)

# Predict and evaluate (simple approach)
predicted_scores = rank_svm.decision_function(features.values)
predicted_ranking = np.argsort(predicted_scores)[::-1]

# Evaluate ranking (NDCG, precision at k, etc.)
# Placeholder for evaluation metric implementation

# Example integration (weighted ranking)
combined_scores = 0.5 * regressor.predict(features.values) + 0.5 * rank_svm.decision_function(features.values)
final_ranking = np.argsort(combined_scores)[::-1]

print(data.iloc[final_ranking].head(10))  # Top 10 ranked clips
