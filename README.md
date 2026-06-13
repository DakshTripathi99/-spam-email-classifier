# Spam Email Classifier using Machine Learning

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using classic NLP techniques and supervised learning algorithms.

---

## Project Overview

Spam messages are a persistent problem in digital communication. This project builds and evaluates machine learning models capable of automatically detecting spam SMS messages based on their textual content.

The pipeline covers the complete ML workflow:
- Text cleaning and preprocessing
- Feature extraction using TF-IDF
- Model training using Multinomial Naive Bayes and Logistic Regression
- Performance evaluation using Accuracy, F1 Score, and Confusion Matrix

---

## Dataset

**Dataset Used:** SMS Spam Collection Dataset  
**Size:** ~5,500 labeled SMS messages  
**Labels:**
- `ham` → Legitimate message
- `spam` → Unwanted/promotional message

Each row in the dataset contains:
| Column | Description |
|--------|-------------|
| label  | "spam" or "ham" |
| message | Raw SMS text |

> See `dataset_link.txt` for the dataset source.

---

## Approach

1. **Text Preprocessing**
   - Convert text to lowercase
   - Remove punctuation and special characters
   - Tokenize text into words
   - Remove stopwords (common English words with no predictive value)

2. **Feature Extraction**
   - Convert cleaned text into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**

3. **Model Training**
   - Split data into training and testing sets (80/20)
   - Train two classification models:
     - **Multinomial Naive Bayes**
     - **Logistic Regression**

4. **Evaluation**
   - Accuracy Score
   - F1 Score
   - Confusion Matrix visualization

---

## Models Used

| Model | Description |
|-------|-------------|
| Multinomial Naive Bayes | Probabilistic model well-suited for text classification with discrete features (word counts/TF-IDF) |
| Logistic Regression | Linear model that estimates probability of a message being spam |

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|-----------|
| Multinomial Naive Bayes | ~96.5% | ~0.91 |
| Logistic Regression | ~96.0% | ~0.92 |

> Note: Exact values may vary slightly depending on train-test split and random seed.

Confusion matrices for both models are displayed when running `model.py`.

---

## Libraries Used

- `pandas` – Data loading and manipulation
- `numpy` – Numerical operations
- `scikit-learn` – ML models, TF-IDF, evaluation metrics
- `matplotlib` – Confusion matrix visualization
- `nltk` – Stopword removal and tokenization

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Download the dataset from the link in `dataset_link.txt`
- Save it as `spam.csv` in the project root folder
- Ensure the CSV has two columns: `label` and `message`

### 4. Run the model
```bash
python model.py
```

The script will output:
- Accuracy and F1 Score for both models
- Confusion matrix plots

---

##  Project Structure
spam-email-classifier/

│
├── README.md          # Project documentation
├── model.py            # Main ML pipeline
├── requirements.txt    # Project dependencies
└── dataset_link.txt     # Dataset source link
---

## Future Improvements

- Add a Streamlit/Flask web app for real-time message prediction
- Experiment with deep learning models (LSTM, BERT)

---

## License

This project is open-source and available for educational purposes.
