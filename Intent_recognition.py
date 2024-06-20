import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re

# Load and preprocess the data
file_path = 'sample_data.csv' 
data = pd.read_csv(file_path)

texts = data['text']
intents = data['intent']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

texts = texts.apply(preprocess_text)

label_encoder = LabelEncoder()
encoded_intents = label_encoder.fit_transform(intents)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_intents, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# st.write(f'Accuracy: {accuracy:.2f}')

# unique_classes = label_encoder.inverse_transform(sorted(set(y_test)))
# st.write(f'Unique classes in y_test: {unique_classes}')

def predict_intent(text, model, vectorizer, label_encoder):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vectorized)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Streamlit app code
st.title('Intent Prediction App')

st.write("""
This app predicts the intent of a given text input using a Random Forest model.
""")

user_input = st.text_input("Enter your query here:")
if st.button('Predict'):
    if user_input:
        predicted_intent = predict_intent(user_input, rf, vectorizer, label_encoder)
        st.write(f'The predicted intent for "{user_input}" is: {predicted_intent}')
    else:
        st.write("Please enter a query to get a prediction.")
