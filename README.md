# Spam Email Classifier 📩🚫

A Machine Learning project that classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) techniques.

This project was built using Python and trained on the **SMS Spam Collection Dataset** from Kaggle.

---

## 📌 Project Overview

Spam messages are unwanted messages that often contain advertisements, scams, or malicious links.
This project builds a **machine learning model** that can automatically detect spam messages.

The model analyzes the **text content of SMS messages** and predicts whether the message is spam or not.

---

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Natural Language Processing (NLP)
* CountVectorizer
* Naive Bayes Algorithm

---

## 📊 Dataset

Dataset used: **SMS Spam Collection Dataset**

Dataset contains:

* **5572 SMS messages**
* Two labels:

  * `ham` → Not spam
  * `spam` → Spam message

Example:

| Message               | Label |
| --------------------- | ----- |
| Win a free iPhone now | spam  |
| Let's meet tomorrow   | ham   |

---

## ⚙️ How the Model Works

1. Load dataset using **Pandas**
2. Clean and preprocess the text data
3. Convert text into numerical features using **CountVectorizer**
4. Split dataset into **training and testing data**
5. Train a **Multinomial Naive Bayes model**
6. Evaluate the model using **accuracy score**
7. Allow users to test the model with their own messages

---

## 📈 Model Performance

* Algorithm: **Multinomial Naive Bayes**
* Accuracy: **~95–98%** depending on training split

---

## 📂 Project Structure

```
Spam_Email_Classifier
│
├── spam_classifier.py
├── spam.csv
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/spam-mail-classification.git
```

### 2️⃣ Install dependencies

```
pip install pandas scikit-learn
```

### 3️⃣ Run the program

```
python spam_classifier.py
```

---

## 💬 Example Usage

```
Model Accuracy: 0.97

Test the Spam Classifier
Enter a message: Win free money now
Prediction: Spam
```

---

## 🚀 Future Improvements

* Add a **Streamlit web interface**

---

## 👨‍💻 Author

Sai Koushik
Artificial Intelligence & Machine Learning Student

---

## ⭐ If you like this project

Give this repository a **star ⭐ on GitHub**.
