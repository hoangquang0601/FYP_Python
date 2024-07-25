import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Sentiment Analysis App")

# User input
user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    if user_input:
        # Transform the user input using the vectorizer
        user_input_tfidf = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(user_input_tfidf)
        
        # Map prediction to sentiment label
        sentiment_mapping = {0: 'NEG', 1: 'NEU', 2: 'POS'}
        result = sentiment_mapping[prediction[0]]
        
        st.write(f"Predicted Sentiment: {result}")
    else:
        st.write("Please enter a comment to analyze.")
