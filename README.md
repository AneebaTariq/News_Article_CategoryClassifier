# News Article Topic Classifier

A machine learning project for automatically classifying news articles into four topic categories: **Business, Entertainment, Health, and Technology**.

This project was developed as part of my Data Science internship at **SAFEX Solutions** and demonstrates an end-to-end machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, hyperparameter tuning, and deployment.

## Project Objective

The goal of this project is to build a predictive model that can determine the topic of a news article based on its title and article content.

## Dataset

The project uses the **UCI News Aggregator Dataset**, containing news articles collected from various sources.

The dataset contains four topic categories:

- `b` — Business
- `e` — Entertainment
- `m` — Health
- `t` — Technology

The dataset used in this project contains **10,348 articles** and **10 original features**.

## Project Workflow

The project follows an end-to-end machine learning pipeline:

1. Dataset exploration
2. Data cleaning
3. Feature engineering
4. Text preprocessing
5. TF-IDF feature extraction
6. Train-test split
7. Model training
8. Model comparison
9. Hyperparameter tuning
10. Confusion matrix and error analysis
11. Feature importance analysis
12. Streamlit deployment

## Feature Engineering

Five additional features were created from the article text:

- Title word count
- Title character count
- Average title word length
- Number of digits in the title
- Article content word count

## Machine Learning Models

Three classification algorithms were evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

TF-IDF was used to convert the article text into numerical features.

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.34% | 97.35% | 97.34% | 97.34% |
| Multinomial Naive Bayes | 95.27% | 95.36% | 95.27% | 95.27% |
| Linear SVM | 97.97% | 97.97% | 97.97% | 97.97% |
| Tuned Linear SVM | 97.83% | 97.83% | 97.83% | 97.83% |

The Linear SVM was selected for hyperparameter tuning using `GridSearchCV`. Five values of `C` were tested using 5-fold cross-validation and weighted F1 scoring. The selected parameter was `C = 0.5`.

## Error Analysis

The tuned model misclassified **45 out of 2,070 test articles**.

The largest source of confusion was between the **Business** and **Technology** categories. Articles involving companies, automobiles, cryptocurrency, financial activity, and technology products sometimes contained vocabulary associated with both categories.

## Feature Importance

The Linear SVM learned meaningful topic-specific vocabulary.

Examples of important features include:

- **Business:** bank, insurance, stock, trading
- **Entertainment:** film, music, movie, actor
- **Health:** health, cancer, disease, treatment
- **Technology:** Google, Apple, Microsoft, Xbox

## Streamlit Demo

The project includes an interactive Streamlit application that allows users to enter a news article title and its content and receive a predicted topic.

The application uses the trained TF-IDF vectorizer, scaler, and tuned Linear SVM model.

**Live Demo:** [Add your Streamlit link here]

## Project Files

```text
news article classification/
│
├── news_classification.ipynb

├── app.py
├── svm_model.pkl
├── tfidf.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
