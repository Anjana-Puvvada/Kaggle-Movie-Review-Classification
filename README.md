# Classification of Kaggle Movie Reviews

## Project Description
This project aims to classify and predict the sentiment of movie reviews using the Kaggle movie reviews dataset. The sentiment classification scale ranges from 0 to 4, where:
- **0**: Negative
- **1**: Strong Negative
- **2**: Neutral
- **3**: Positive
- **4**: Strong Positive

The dataset is based on movie reviews from Rotten Tomatoes, annotated for sentiment analysis.

## Key Components

### Data Description
- **Training Data**: 156,060 entries with columns `PhraseId`, `SentenceId`, `Phrase`, and `Phrase Sentiment`.
- **Testing Data**: 66,292 entries with columns `PhraseId`, `SentenceId`, and `Phrase`.

### Data Preprocessing
- Reading and loading data into the Python environment.
- Tokenization and filtering of phrases.
- Pre-processing steps include:
  - Removing punctuation.
  - Converting text to lowercase.
  - Removing stop words.

### Feature Engineering
- **Bag of Words (BOW)/Unigram Features**: Creating lists of most frequent words.
- **Sentiment Lexicons**: Using positive and negative word counts from a lexicon.
- **Negation Word Features**: Handling negation words to accurately capture sentiment.
- **Part-of-Speech (POS) Features**: Using POS tags to identify syntactic patterns.
- **Bigram and Trigram Features**: Creating high-frequency bigram and trigram features.
- **Combined Feature Sets**: Combining multiple feature sets for improved accuracy.

### Classification Algorithms
- **Naïve Bayes Classifier (NLTK)**: Implemented with various feature sets (unigrams, bigrams, trigrams, POS, etc.).
- **Logistic Regression (Scikit-Learn)**: Utilized with different feature sets and evaluated for precision, recall, and F1-score.
- **Support Vector Machines (Scikit-Learn)**: Applied to different feature sets and analyzed for performance.
- **Decision Tree Classifier (Scikit-Learn)**: Used with various feature sets to classify sentiments.

### Results and Observations
- Analysis of classifier performance using different feature sets.
- Evaluation metrics include accuracy, precision, recall, and F1-score.
- Combined feature sets generally yielded the highest accuracy.

### Advanced Tasks and Summary
- Additional experiments and advanced tasks to improve model performance.
- Summary of experiments and key observations.

## Lessons Learned
- Importance of feature engineering in improving model performance.
- Challenges with pre-processing and its impact on model accuracy.
- Benefits of combining multiple features to capture various aspects of data.

## Repository Structure
- **Data**: Contains the training and testing datasets.
- **Notebooks**: Jupyter notebooks for data preprocessing, feature engineering, and classification experiments.
- **Scripts**: Python scripts for implementing various classifiers and feature extraction methods.
- **Results**: CSV files and logs of model performance metrics.
- **Reports**: Documentation and reports summarizing the project, methodology, and results.

This project demonstrates a comprehensive approach to sentiment analysis of movie reviews using various NLP techniques and classification algorithms. The repository provides all necessary code, data, and documentation to understand and replicate the analysis.
