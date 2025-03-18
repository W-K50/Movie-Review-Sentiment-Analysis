**Movie Review Sentiment Analysis Report**

**1. Introduction**
This project implements a sentiment analysis model to classify movie reviews as either positive or negative. The dataset used is the IMDB movie review dataset, which contains labeled reviews. The goal is to preprocess the text, train a Naïve Bayes classifier, and evaluate its performance.

**2. Dataset Overview**
- The dataset contains **50,000** movie reviews with labeled sentiments.
- The sentiment labels are binary: **positive (1)** and **negative (0)**.
- The dataset was split into **80% training data** and **20% testing data**.

**3. Data Preprocessing**
To improve model performance, the following preprocessing steps were applied:
- Converted text to **lowercase**.
- Removed **special characters** and **stopwords**.
- Applied **tokenization**.
- Vectorized text using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

**4. Exploratory Data Analysis (EDA)**
- Visualized sentiment distribution using **count plots**.
- Identified the most frequently used words using **bar plots**.
- Generated a **word cloud** to represent common words in reviews.

**5. Model Training and Evaluation**
- Used **Multinomial Naïve Bayes** classifier for sentiment classification.
- Achieved the following performance on the test dataset:
  - **Accuracy:** 86.7%
  - **F1-score:** 86.5%

**6. Deployment**
- A function was implemented to predict sentiment for new reviews.
- The trained **Naïve Bayes model** and **TF-IDF vectorizer** were saved using **Joblib** for future use.

**7. Conclusion**
This project successfully classifies movie reviews with high accuracy using Naïve Bayes. Future improvements could include experimenting with deep learning models like LSTMs or Transformers to further enhance performance.

