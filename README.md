# Syrian War Fake News Classifier

## Overview
The **Syrian War Fake News Classifier** is a project designed to analyze and classify news articles related to the Syrian war as either fake or real. Using a combination of natural language processing (NLP), machine learning, and deep learning techniques, the project provides a pipeline for preprocessing, visualization, classification, and evaluation of news articles.

---

## Dataset
The project uses the **FA-KES-Dataset.csv**, which contains:
- **`article_title`**: The title of the news article.
- **`labels`**: Indicates the class of the news: `0` for fake and `1` for real.

The dataset is processed to remove irrelevant columns, check for missing values, and clean the text for further analysis.

---

## Features
### 1. **Data Preprocessing**
- Dropping irrelevant columns.
- Handling missing values.
- Cleaning text: removing special characters and unnecessary spaces.

### 2. **Visualizations**
- Class Distribution: Bar plot showing the number of fake vs. real articles.
- Average Word Count: Comparison of average word counts between fake and real news.
- Top Words by Class: Frequent words in fake and real news, visualized with bar plots.
- Word Clouds: Visual representation of common words for both fake and real news.
- News Length Distribution: Histogram showing the distribution of article word counts by class.

### 3. **Machine Learning**
- **Vectorization**: Text data is vectorized using TF-IDF with a maximum feature limit of 5000.
- **Logistic Regression**: A baseline classification model with results including accuracy and a classification report.

### 4. **Deep Learning**
- A sequential neural network is trained using the processed data. The architecture includes:
  - Dense layers with ReLU activation.
  - Dropout layers for regularization.
  - A final sigmoid layer for binary classification.
- Evaluation metrics include precision, recall, F1 score, AUC, and ROC curve visualization.

### 5. **Prediction**
- A function, `predict_news`, takes user input, preprocesses it, and predicts whether the news is real or fake.

---

## Results
- **Logistic Regression**:
  - Accuracy: Displayed with a classification report.
- **Neural Network**:
  - Accuracy: ~XX% (as trained and evaluated).
  - Metrics: Precision, recall, F1 score, and AUC.
  - Visualization: ROC curve for performance analysis.

---

## How to Use
1. Clone the repository or download the script.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script to load the dataset and train models:
   ```bash
   python syrian_war_article_classifier.py
   ```
4. Enter a news title when prompted to classify it.

---

## Dependencies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- WordCloud

---

## Future Work
- Expand the dataset for improved generalizability.
- Explore advanced NLP models such as BERT or GPT for enhanced classification.
- Deploy the model as a web application for real-time prediction.

---

## Acknowledgments
This project was built using the FA-KES-Dataset.csv and incorporates various open-source libraries and frameworks.

