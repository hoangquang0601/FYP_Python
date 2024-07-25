import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load your DataFrame
df = pd.read_csv('Skincare_Clean_Data_Before_Labelling.csv')

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to get sentiment label
def get_sentiment_label(text):
    try:
        # Ensure text is not empty or too short
        if not text or len(text.strip()) == 0:
            return None
        result = sentiment_analysis(text)
        label = result[0]['label']
        return label
    except Exception as e:
        print(f"Error processing text: '{text}' -> {e}")
        return None

# Apply the sentiment analysis with progress bar
tqdm.pandas(desc="Labelling Comments")
df['Sentiment Label'] = df['Comment Text'].progress_apply(get_sentiment_label)

# Save the labelled DataFrame
df.to_csv('labelled_data.csv', index=False)

# Display the first few rows to verify
print(df.head())
