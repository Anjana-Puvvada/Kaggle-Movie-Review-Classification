
#import sentiment_read_subjectivity
import os
import sys
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
import re
#import sentiment_read_subjectivity
from classifyKaggle import create_test_submission, get_word_features, usual_features, preprocess_document, get_words_from_test_dataset



# Assuming the other functions (pre_processing_documents, SL_features, etc.) are already defined above this script.

def process_dataset(filepath, limit=None, preprocess=False):
    """Process the dataset to form a list of document tuples (tokens, label)."""
    with open(filepath, 'r') as file:
        data = []
        for line in file:
            if not line.startswith('Phrase'):
                line = line.strip()
                parts = line.split('\t')
                if len(parts) == 4:
                    text, label = parts[2], int(parts[3])
                    if preprocess:
                        tokenizer = RegexpTokenizer(r'\w+')
                        text = pre_processing_documents(text)
                        tokens = tokenizer.tokenize(text)
                    else:
                        tokens = nltk.word_tokenize(text)
                    data.append((tokens, label))
        if limit:
            random.shuffle(data)
            data = data[:limit]
    return data

def create_featuresets(phrasedocs, feature_extraction_function, *args):
    """Create feature sets using the provided feature extraction function."""
    # Make sure to unpack the tuple (document, label)
    return [(feature_extraction_function(doc, *args), label) for (doc, label) in phrasedocs]

def accuracy_calculation(featuresets):
    # Splitting the dataset into training and testing sets
    training_size = int(len(featuresets) * 0.8)  # Using 80% for training and 20% for testing
    training_set, test_set = featuresets[:training_size], featuresets[training_size:]

    # Training the classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    
    # Calculating accuracy
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print("Accuracy of classifier:", accuracy)
    print("---------------------------------------------------")
    
    # Displaying the most informative features
    classifier.show_most_informative_features(10)

    # Generating and printing the confusion matrix
    print_confusionmatrix(classifier, test_set)

def print_confusionmatrix(classifier, test_set):
    # Prepare data for the confusion matrix
    actual = [label for (features, label) in test_set]
    predicted = [classifier.classify(features) for (features, label) in test_set]
    
    # Create the confusion matrix
    cm = nltk.ConfusionMatrix(actual, predicted)
    print("\nThe confusion matrix")
    print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=9))

def main(training_path, test_path, limit=None):
    # Process training and test datasets
    training_docs = process_dataset(training_path, limit, preprocess=True)
    test_docs = process_dataset(test_path, preprocess=True)

    # Extract features
    all_words_from_docs = [doc for doc, _ in training_docs]  # Ensure only document parts are passed
    word_features = get_word_features(get_words_from_phasedocs(all_words_from_docs))
    training_set = create_featuresets(training_docs, normal_features, word_features)
    test_set = create_featuresets(test_docs, normal_features, word_features)

    # Combine training and test sets to simulate a complete experiment
    combined_set = training_set + test_set
    accuracy_calculation(combined_set)


if __name__ == '__main__':
    dirPath = '/Users/anjanapuvvada/Desktop/NLP_Finalproject_Kaggle/corpus'
    training_file = os.path.join(dirPath, 'train.tsv')
    test_file = os.path.join(dirPath, 'test.tsv')
    main(training_file, test_file)






