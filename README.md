# Sentiment Analysis on Movie Reviews

## Project Overview
This project involved conducting sentiment analysis on movie reviews using multiple machine learning classifiers. The primary goal was to classify reviews as positive or negative by leveraging both linguistic and sentiment-specific features. The project achieved high accuracy by combining Sentiment Lexicon (SL) and LIWC (Linguistic Inquiry and Word Count) features, offering insights into text-based sentiment classification.

---

## Goals
- Perform sentiment analysis on movie reviews using machine learning classifiers.
- Engineer a comprehensive feature set, including linguistic and sentiment-based features, for improved model performance.
- Evaluate and compare the performance of multiple classifiers.
- Visualize sentiment trends and word usage patterns for better interpretability.

---

## Methods
### Data Preprocessing
- **Lowercase Conversion**: Converted all text to lowercase for consistency.
- **Punctuation Removal**: Removed punctuation marks to standardize tokenization.
- **Token Filtering**: Filtered out stop words and irrelevant tokens to reduce noise.

### Feature Engineering
- **N-grams**: Generated unigrams and bigrams to capture contextual patterns in text.
- **POS Tags**: Extracted Part-of-Speech (POS) tags to analyze linguistic structure.
- **Sentiment Lexicon Features**: Incorporated features from established sentiment lexicons.
- **LIWC Features**: Integrated features from the LIWC tool to analyze psychological and emotional aspects of text.

### Machine Learning Models
- **Naive Bayes**
- **Decision Tree**
- **Support Vector Machine (SVM)**
- **Random Forest**

### Evaluation
- Performed cross-validation to ensure robust model evaluation.
- Assessed models using metrics like accuracy, precision, recall, and F1-score.

---

## Results
### Model Performance
The combination of Sentiment Lexicon (SL) and LIWC features significantly enhanced classification accuracy across all models. Key insights include:
- **Best Model**: Random Forest achieved the highest accuracy, followed closely by SVM.
- **Feature Importance**: Sentiment Lexicon features were the most influential, followed by unigrams and bigrams.

### Visualizations
- **Sentiment Distribution**: Visualized the distribution of positive and negative reviews in the dataset.
- **Word Frequency Charts**: Displayed the most frequent unigrams and bigrams for positive and negative reviews, providing insights into common word usage.

---

## Achievements
- Successfully applied and compared multiple classifiers for sentiment analysis.
- Engineered diverse features, combining linguistic and sentiment-based aspects, for improved model performance.
- Gained hands-on experience in Natural Language Processing (NLP) and feature engineering.
- Visualized sentiment patterns and word usage trends for interpretability.

---

## Future Work
- Experiment with deep learning approaches such as LSTM or BERT for enhanced performance.
- Incorporate additional features like topic modeling or semantic embeddings.
- Explore real-time sentiment analysis on streaming data sources.

---

## Authors
- **Resham Bahira**

