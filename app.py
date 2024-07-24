import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the model and vectorizer
model_file_path = 'sentiment_model.pkl'  # Adjust the path as needed
vectorizer_file_path = 'vectorizer.pkl'  # Adjust the path as needed

model = joblib.load(model_file_path)
vectorizer = joblib.load(vectorizer_file_path)

# Define stop words
stop_words = set(stopwords.words('english'))

# Preprocessing function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app
st.title("Sentiment Analysis App")

st.write("This app uses a logistic regression model to predict the sentiment of your text.")

# Input text
input_text = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if input_text:
        # Preprocess the input text
        cleaned_text = clean_text(input_text)
        
        # Vectorize the input text
        input_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict the sentiment
        prediction = model.predict(input_vectorized)[0]
        
        # Display the result
        st.write(f"The sentiment of the text is: **{prediction}**")
    else:
        st.write("Please enter some text to analyze.")

# Run the app using the following command:
# streamlit run app.py
