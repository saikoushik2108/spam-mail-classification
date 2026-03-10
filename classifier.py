import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Features and labels
X = df['message']
y = df['label']

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# ---- Test with custom message ----
print("\nTest the Spam Classifier")
message = input("Enter a message: ")

message_data = vectorizer.transform([message])
prediction = model.predict(message_data)

if prediction[0] == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Not Spam")