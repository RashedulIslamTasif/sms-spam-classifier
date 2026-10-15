# SMS Spam Classification using SVM & Naive Bayes

A machine learning project that classifies SMS text messages as **spam** or **ham** (legitimate) using natural language processing and classic ML classifiers.

## Overview

This project builds and compares two text classification models — **Support Vector Machine (LinearSVC)** and **Multinomial Naive Bayes** — to automatically detect spam messages in SMS text data. It covers the full pipeline: data cleaning, exploratory data analysis, text preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation.

## Dataset

- **Source:** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 messages
- **Labels:** `ham` (legitimate) or `spam`
- **Format:** CSV with message text and label columns

## Project Workflow

1. **Data Loading & Cleaning**
   - Loaded the raw CSV and selected the relevant columns
   - Renamed columns to `label` and `message`
   - Removed duplicate entries

2. **Exploratory Data Analysis (EDA)**
   - Checked class distribution (ham vs. spam)
   - Analyzed message length and word count
   - Generated word clouds to visualize common terms in spam vs. ham messages

3. **Text Preprocessing**
   - Lowercased all text
   - Removed URLs, email addresses, numbers, and punctuation
   - Removed stopwords and applied lemmatization

4. **Feature Extraction**
   - Converted cleaned text into numerical features using **TF-IDF vectorization**

5. **Model Training**
   - **SVM (LinearSVC)** — trained via a `Pipeline` combining TF-IDF vectorization and a linear SVM classifier

6. **Hyperparameter Tuning**
   - Used `GridSearchCV` to tune the SVM's `C` parameter
   - Best parameters: `C = 1`
   - Best cross-validation score: **~97.9%**

7. **Model Evaluation**
   - Evaluated using accuracy, precision, recall, F1-score, and confusion matrix
   - Performed 5-fold cross-validation to confirm model stability (mean accuracy ~97.9%)

8. **Custom Predictions**
   - Tested both models on custom example messages to verify real-world behavior

## Results

| Model | Cross-Validation Accuracy |
|---|---|
| SVM (LinearSVC, tuned) | ~97.9% |


## Tech Stack

- **Python 3**
- **pandas**, **numpy** — data handling
- **matplotlib**, **seaborn**, **wordcloud** — visualization
- **nltk** — text preprocessing (stopwords, lemmatization)
- **scikit-learn** — TF-IDF vectorization, SVM, Naive Bayes, GridSearchCV, evaluation metrics

## How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/<your-username>/sms-spam-classifier.git
   cd sms-spam-classifier
   ```
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
   ```
3. Open `SMS_Spam_SVM.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab
4. Run all cells in order from top to bottom

## Project Structure

```
sms-spam-classifier/
├── SMS_Spam_SVM.ipynb    # Main notebook: full pipeline and analysis
├── README.md              # Project documentation
└── spam.csv                # Dataset (optional — or link to Kaggle instead)
```

## Acknowledgments

Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) via [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
