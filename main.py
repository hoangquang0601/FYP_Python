# read the Skincare_Labelling_Data_Preprocessing_Before_Translation.csv file
import pandas as pd
import numpy as np

df = pd.read_csv('Skincare_Labelling_Data_Preprocessing_Before_Translation.csv')

from googletrans import Translator
from tqdm import tqdm

translator = Translator()

def translate_to_english_with_progress(text):
    try:
        # Translate to English
        translated = translator.translate(text, dest='en').text
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

# Apply translation function to the DataFrame
df = translate_df_with_progress(df)

# Display the updated DataFrame
print(df.head())