# News Article Topic Classification

A machine learning project that classifies news articles into four topic categories: **Business, Entertainment, Health, and Technology**.

The project was developed as part of the **SAFEX Solutions Data Science Internship – Week 4 Advanced Predictive Modeling Task**.

## 🚀 Live Demo

Try the deployed Streamlit application:

**Live Demo:** [News Article Topic Classifier](https://newsarticlecategoryclassifier.streamlit.app/)

The application allows users to enter a news article title and content and receive a predicted topic.

---

## 📌 Project Objective

The objective of this project is to build an end-to-end machine learning pipeline for automatically classifying news articles into predefined topic categories.

The project covers:

- Data exploration and cleaning
- Text preprocessing
- Feature engineering
- TF-IDF feature extraction
- Machine learning model training
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Error analysis
- Feature importance analysis
- Streamlit deployment

---

## 📊 Dataset

The project uses the **UCI News Aggregator Dataset**, a public news dataset containing articles from different publishers.

The dataset contains:

- **10,348 news articles**
- **10 original columns**
- **4 topic categories**

### Categories

| Category Code | Topic |
|---|---|
| `b` | Business |
| `e` | Entertainment |
| `m` | Health |
| `t` | Technology |

The original dataset contains information such as article title, URL, publisher, category, story ID, hostname, timestamp, and article content.

---

## 🔍 Data Exploration

The dataset was examined for:

- Dataset shape
- Data types
- Missing values
- Duplicate records
- Empty text fields
- Target/category distribution

### Data Quality

- Rows: **10,348**
- Columns: **10**
- Missing values: **0**
- Exact duplicate rows: **0**
- Duplicate titles: **24**

The duplicate titles were retained because they did not represent exact duplicate records and could belong to different articles or publishers.

---

## 🛠️ Feature Engineering

Several additional features were created from the article title and content.

The engineered features include:

1. **Title Word Count**  
   Number of words in the article title.

2. **Title Character Count**  
   Number of characters in the title.

3. **Title Average Word Length**  
   Average number of characters per title word.

4. **Title Digit Count**  
   Number of numerical digits appearing in the title.

5. **Content Word Count**  
   Number of words in the article content.

The title and article content were also combined into a single text feature for classification.

---

## 🧹 Text Preprocessing

The text data was processed before applying machine learning.

The preprocessing steps included:

- Converting text to lowercase
- Removing unnecessary whitespace
- Combining article title and content
- TF-IDF vectorization
- Creating unigram and bigram features
- Removing English stop words

The TF-IDF vectorizer was configured with a maximum of **30,000 features**.

---

## 🤖 Machine Learning Models

Three classification models were evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.34% | 97.35% | 97.34% | 97.34% |
| Multinomial Naive Bayes | 95.27% | 95.36% | 95.27% | 95.27% |
| Linear SVM | 97.97% | 97.97% | 97.97% | 97.97% |
| Tuned Linear SVM | 97.83% | 97.83% | 97.83% | 97.83% |

The Linear SVM was selected for hyperparameter tuning because it achieved the highest performance among the initial models.

---

## 🎯 Hyperparameter Tuning

`GridSearchCV` with **5-fold cross-validation** was used to tune the Linear SVM.

The following values of `C` were tested:

```text
0.1
0.5
1
2
5
````

The best parameter selected through cross-validation was:

```text
C = 0.5
```

The cross-validation scoring metric was **weighted F1 score**.

The tuned model achieved:

* Accuracy: **97.83%**
* Precision: **97.83%**
* Recall: **97.83%**
* Weighted F1: **97.83%**

The tuned test score was slightly lower than the original Linear SVM score. The tuned model was retained because the hyperparameter was selected using cross-validation rather than the test set.

---

## 📈 Confusion Matrix

The tuned Linear SVM correctly classified **2,025 out of 2,070** test articles.

There were **45 misclassified articles**, giving an error rate of approximately **2.17%**.

The largest number of classification errors occurred between the **Business** and **Technology** categories.

---

## 🔎 Error Analysis

Several misclassified articles were manually examined.

Some articles contained vocabulary associated with multiple topics. For example:

* General Motors articles could contain both business and technology-related terminology.
* Cryptocurrency articles could contain both financial and technology-related concepts.
* Entertainment articles involving streaming platforms could contain technology-related terminology.
* Health articles involving companies and stocks could contain business-related vocabulary.

These examples demonstrate that some news articles naturally overlap multiple topics, making single-category classification more difficult.

---

## ⭐ Feature Importance

The Linear SVM coefficients were examined to identify important words associated with each category.

Examples of important features included:

### Business

```text
bank
insurance
stock
trading
economic
```

### Entertainment

```text
film
music
movie
singer
actor
```

### Health

```text
health
cancer
disease
patients
treatment
```

### Technology

```text
Google
Apple
Microsoft
Xbox
science
```

These features show that the model learned meaningful vocabulary patterns associated with the different news topics.

---

## 🌐 Streamlit Application

A Streamlit web application was created to demonstrate the trained model.

Users can:

1. Enter a news article title.
2. Paste the article content.
3. Click **Predict Topic**.
4. Receive the predicted category.

Example:

```text
Article Title:
Apple unveils new iPhone with advanced features

Predicted Topic:
Technology
```

The application uses the trained Linear SVM model together with the saved TF-IDF vectorizer and feature scaler.

---

## 📁 Project Structure

```text
news-article-categoryclassifier/
│
├── app.py
├── news_classification.ipynb
├── svm_model.pkl
├── tfidf.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                        | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `news_classification.ipynb` | Complete data science and machine learning workflow |
| `app.py`                    | Streamlit web application                           |
| `svm_model.pkl`             | Trained Linear SVM model                            |
| `tfidf.pkl`                 | Trained TF-IDF vectorizer                           |
| `scaler.pkl`                | Feature scaler                                      |
| `requirements.txt`          | Python dependencies                                 |
| `README.md`                 | Project documentation                               |

---

## 💻 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Joblib
* Matplotlib
* Jupyter Notebook
* Streamlit
* Google Colab
* GitHub

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/AneebaTariq/News_Article_CategoryClassifier
```

Navigate to the project directory:

```bash
cd News_Article_CategoryClassifier
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📌 Key Findings

* The Linear SVM achieved **97.97% accuracy** before tuning.
* The tuned Linear SVM achieved **97.83% accuracy** on the unseen test set.
* The model successfully learned topic-specific vocabulary patterns.
* Business and Technology produced the most noticeable classification overlap.
* Some classification errors resulted from articles containing information related to multiple topics.
* The final model provides a practical demonstration of automated news-topic classification.

---

## 🎓 Internship Context

This project was completed as part of the:

**SAFEX Solutions – Data Science Internship**
**Week 4: Advanced Predictive Model – News Article Topic Classification**

The project demonstrates an end-to-end machine learning workflow from data preparation and feature engineering through model development, evaluation, and deployment.

---

## 👤 Author

**Aneeba Tariq**

Data Science Intern
SAFEX Solutions


```markdown
**Live Demo:** [News Article Topic Classifier](https://newsarticlecategoryclassifier.streamlit.app/)
````


```markdown
[git clone News_Article_CategoryClassifier](https://github.com/AneebaTariq/News_Article_CategoryClassifier)
```


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

**Live Demo:** [(https://newsarticlecategoryclassifier.streamlit.app/)]

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
