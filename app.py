import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("Spam Message Classifier")

st.write("Enter a message below to check if it is Spam or Not Spam.")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Features
X = df['message']
y = df['label']

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# User input
user_message = st.text_input("Type your message")

if st.button("Predict"):

    message_vector = vectorizer.transform([user_message])
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        st.error("🚨 This message is Spam!")
    else:
        st.success("✅ This message is Not Spam")