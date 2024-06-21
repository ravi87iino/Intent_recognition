import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
from bs4 import BeautifulSoup


def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    contractions = { 
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
        "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
        "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
        "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
        "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
        "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
        "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
        "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
        "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
        "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "that'd": "that would",
        "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have",
        "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
        "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
        "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
        "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
        "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
        "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
        "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
        "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
        "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
        "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
        "you're": "you are", "you've": "you have"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    q = BeautifulSoup(q, "html.parser").get_text()
   
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


file_path = 'sample_data.csv'
data = pd.read_csv(file_path)

# Ensure there are no missing values in the required columns
data.dropna(subset=['text', 'intent'], inplace=True)

texts = data['text'].apply(preprocess)
intents = data['intent']

label_encoder = LabelEncoder()
encoded_intents = label_encoder.fit_transform(intents)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, encoded_intents, test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

unique_classes = label_encoder.inverse_transform(sorted(set(y_test)))
print(f'Unique classes in y_test: {unique_classes}')

def predict_intent(text, model, vectorizer, label_encoder):
    preprocessed_text = preprocess(text)
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
