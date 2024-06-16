import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re

# Load the data
file_path = 'sample_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Assuming the dataset has columns 'text' for the sentences and 'intent' for the labels
texts = data['text']
intents = data['intent']

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply preprocessing to texts
texts = texts.apply(preprocess_text)

# Encode the labels (intents)
label_encoder = LabelEncoder()
encoded_intents = label_encoder.fit_transform(intents)

# Convert the text data to numerical data using TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_intents, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# # Identify unique classes in y_test and get their corresponding names
unique_classes = label_encoder.inverse_transform(sorted(set(y_test)))
print(f'Unique classes in y_test: {unique_classes}')

# Function to make predictions on new input text
def predict_intent(text, model, vectorizer, label_encoder):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_vectorized)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Example prediction
# new_text = 'hello how are you'
new_text = input("Enter Your Query: ")
predicted_intent = predict_intent(new_text, rf, vectorizer, label_encoder)
print(f'The predicted intent for "{new_text}" is: {predicted_intent}')
