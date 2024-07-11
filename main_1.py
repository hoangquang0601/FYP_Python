import pandas as pd
from deep_translator import GoogleTranslator  # You can also use LibreTranslator, YandexTranslator, etc.
from tqdm import tqdm

df = pd.read_csv('Skincare_Labelling_Data_Preprocessing_Before_Translation - Copy.csv')

# Initialize the translator (using LibreTranslate in this example)
translator = GoogleTranslator(source='auto', target='en')

def translate_to_english_with_progress(text):
    try:
        # Translate to English
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

def translate_df_with_progress(df):
    # Initialize tqdm with the length of the DataFrame
    tqdm.pandas(desc="Translating Comments")
    # Apply translation function with progress bar
    df['Comment Text'] = df['Comment Text'].progress_apply(translate_to_english_with_progress)
    return df

df = translate_df_with_progress(df)

# Display the updated DataFrame
print(df.head())
