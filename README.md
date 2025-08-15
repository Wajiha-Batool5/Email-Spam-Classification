# 📧 Email Spam Classification

A machine learning project that detects spam emails using **TF-IDF**, **SMOTE**, and multiple classification algorithms (**Naive Bayes, Logistic Regression, SVM**).
This project includes data preprocessing, feature engineering, model training with hyperparameter tuning, and visual performance analysis.

---

## 🚀 Features

* **Data Cleaning & Preprocessing**

  * Lowercasing, punctuation & number removal
  * Stopword removal
  * Stemming (Porter Stemmer)
  * URL & numeric token replacement
* **Feature Extraction**

  * TF-IDF with unigrams & bigrams
  * Top 5000 features for performance
* **Class Balancing**

  * **SMOTE** for handling imbalance between spam and ham
* **Model Training**

  * Naive Bayes
  * Logistic Regression
  * Support Vector Machine (LinearSVC)
* **Evaluation**

  * Accuracy, Precision, Recall, F1-score
  * Confusion Matrices
  * Model comparison charts
* **Model Saving & Reuse**

  * Save models & vectorizer for future predictions
* **Custom Predictions**

  * Test with your own email text

---

## 📂 Project Structure

```
Email-Spam-Classification-main/
│
├── spam.csv                          # Dataset
├── Email Spam Classification.ipynb   # Main notebook
├── README.md                          # Documentation
├── requirements.txt                   # Dependencies
├── saved_models/                      # Saved models & vectorizer 
└── screanshots/                      # Images 
```

---

## 🛠 Installation

1. **Clone this repository**

```bash
git clone https://github.com/Wajiha-Batool5/Email-Spam-Classification.git
cd Email-Spam-Classification-main
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download NLTK stopwords**

```python
import nltk
nltk.download('stopwords')
```

---

## 📜 Requirements (`requirements.txt`)

```
pandas
numpy
nltk
scikit-learn
imbalanced-learn
matplotlib
joblib
```

---

## 📊 Usage

### Run the notebook

Open **Email Spam Classification.ipynb** and run all cells to:

* Load & preprocess data
* Train and evaluate models
* Visualize performance
* Save models

### Example Prediction

```python
import joblib
from preprocessing import clean_text  # If function is separated

vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('svm_model.joblib')  # Load your best model

email = ["Free money!!! Click here now"]
cleaned_email = [clean_text(text) for text in email]
email_vector = vectorizer.transform(cleaned_email)
print(model.predict(email_vector))
```

---

## 📈 Results

**Best Model:** SVM (based on F1-score for spam)

| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Naive Bayes         | 0.97     | 0.96      | 0.94     | 0.95     |
| Logistic Regression | 0.98     | 0.97      | 0.96     | 0.96     |
| SVM                 | **0.99** | **0.98**  | **0.98** | **0.98** |

---

## 📷 Screenshots

**Model Comparison Chart:**

<img width="220" height="565" alt="confusion_matrices" src="https://github.com/user-attachments/assets/bec8c7d8-8fa7-4a13-b6b6-2d8c644c3020" />


**Confusion Matrices Example:**

<img width="1232" height="710" alt="model_comparison" src="https://github.com/user-attachments/assets/7ea7addb-e713-471d-ba98-5005a906bfd5" />


---

## 📌 Dataset

* **File:** `spam.csv`
* **Columns:**

  * `Category`: `ham` or `spam`
  * `Message`: Email/SMS text

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit your changes
4. Open a pull request
