import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import isodate

# Function to convert ISO 8601 duration to seconds
def duration_to_seconds(duration):
    return isodate.parse_duration(duration).total_seconds()

# Load data
minecraft_df = pd.read_csv("minecraft.csv")
valorant_df = pd.read_csv("valorant.csv")

# Combine data into a single dataframe
df = pd.concat([minecraft_df, valorant_df])

# Preprocess data
df['duration'] = df['duration'].apply(duration_to_seconds)
df['views'] = df['views'].astype(int)
df['likes'] = df['likes'].astype(int)
df['comments'] = df['comments'].astype(int)

# Define virality (example: views >= 100000 as viral)
df['viral'] = np.where(df['views'] >= 100000, 1, 0)

# Feature engineering
df['likes_per_view'] = df['likes'] / df['views']
df['comments_per_view'] = df['comments'] / df['views']

# Fill NaN values resulting from division by zero
df.fillna(0, inplace=True)

# Select features and target
features = ['duration', 'views', 'likes', 'comments', 'likes_per_view', 'comments_per_view']
target = 'viral'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = LogisticRegression(max_iter=1000, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Cross-validation scores
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_model, 'logistic_virality_model.joblib')
