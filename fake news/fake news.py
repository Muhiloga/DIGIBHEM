import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.stem import SnowballStemmer

# Download stopwords and initialize stemmer
nltk.download('stopwords')
stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")

# Load datasets
true_news = pd.read_csv("News _dataset/True.csv")
fake_news = pd.read_csv("News _dataset/Fake.csv")

# Add labels for binary classification (1 = Real, 0 = Fake)
true_news["label"] = 1
fake_news["label"] = 0

# Combine both datasets
news_data = pd.concat([true_news, fake_news], axis=0)

# Shuffle the dataset
news_data = news_data.sample(frac=1).reset_index(drop=True)

# Basic text cleaning function
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\d+', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove stopwords and apply stemming
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    
    return text

# Apply text preprocessing
news_data['cleaned_text'] = news_data['text'].apply(preprocess_text)

# Separate features and labels
X = news_data['cleaned_text']
y = news_data['label']
# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
