import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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

# Create TF-IDF vectorizer and SVM pipeline
pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))

# Train the model with progress bar
with tqdm(total=len(X_train), desc="Training SVM") as pbar:
    pipeline.fit(X_train, y_train)
    pbar.update(len(X_train))

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
