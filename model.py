"""
Spam Email Classifier using Machine Learning
-----------------------------------------------
This script:
1. Loads the SMS Spam Collection dataset
2. Preprocesses the text (lowercasing, tokenization, stopword removal)
3. Converts text to numerical features using TF-IDF
4. Trains Multinomial Naive Bayes and Logistic Regression models
5. Evaluates both models using Accuracy, F1 Score, and Confusion Matrix
"""

import pandas as pd
import numpy as np
import string
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Download stopwords (only needs to run once)
nltk.download('stopwords')
nltk.download('punkt')


# ----------------------------
# STEP 1: Load Dataset
# ----------------------------
def load_dataset(filepath="spam.csv"):
    """
    Loads the SMS Spam Collection dataset.
    Expects a CSV file with columns: 'label' and 'message'
    """
    df = pd.read_csv(filepath, encoding="latin-1")

    # Keep only the relevant columns (some versions of the dataset have extra columns)
    df = df[['label', 'message']]

    # Drop any rows with missing values
    df.dropna(inplace=True)

    return df


# ----------------------------
# STEP 2: Text Preprocessing
# ----------------------------
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Cleans and preprocesses a single text message:
    - Converts to lowercase
    - Removes punctuation
    - Tokenizes into words
    - Removes stopwords
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize (simple whitespace split is sufficient here)
    tokens = text.split()

    # Remove stopwords
    cleaned_tokens = [word for word in tokens if word not in stop_words]

    # Join back into a single string
    return " ".join(cleaned_tokens)


# ----------------------------
# STEP 3: Evaluation Helper
# ----------------------------
def evaluate_model(model_name, y_test, y_pred):
    """
    Prints accuracy, F1 score, and displays a confusion matrix
    for the given model's predictions.
    """
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='spam')

    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    # Load data
    df = load_dataset("spam.csv")

    # Preprocess all messages
    print("Preprocessing text...")
    df['cleaned_message'] = df['message'].apply(preprocess_text)

    # Features and labels
    X = df['cleaned_message']
    y = df['label']

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorization
    print("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ----------------------------
    # Model 1: Multinomial Naive Bayes
    # ----------------------------
    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_predictions = nb_model.predict(X_test_tfidf)
    evaluate_model("Multinomial Naive Bayes", y_test, nb_predictions)

    # ----------------------------
    # Model 2: Logistic Regression
    # ----------------------------
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    evaluate_model("Logistic Regression", y_test, lr_predictions)


if __name__ == "__main__":
    main()
