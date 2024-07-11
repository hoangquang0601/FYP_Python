import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load your dataset
df = pd.read_csv('labelled_data.csv')

# Prepare text data
texts = df['Comment Text']
labels = df['Sentiment Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer and Naive Bayes classifier pipeline
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

# Transform the text data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model with progress bar
for i in tqdm(range(X_train_tfidf.shape[0]), desc="Training Progress"):
    classifier.partial_fit(X_train_tfidf[i:i+1], y_train.iloc[i:i+1], classes=np.unique(y_train))

# Predict and evaluate
y_pred = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
