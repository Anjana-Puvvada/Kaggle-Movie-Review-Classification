import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

def process_data(file_path):
    # Number of folds for cross-validation
    num_folds = 10

    # Read in the dataset using pandas
    dataset = pd.read_csv(file_path)
    
    # Print the shape of the dataset (number of instances, number of features + class label)
    print('Shape of dataset - number of instances with number of features + class label:')
    print(dataset.shape)
    
    # Convert the pandas DataFrame to a numpy array for use with scikit-learn
    data_array = dataset.values
    
    # Extract the last column to use as class labels
    labels = data_array[:, -1]
    
    # Extract the remaining columns as feature matrix
    features = data_array[:, :-1]
    
    # Output results from Logistic Regression
    print('** Results from Logistic Regression **')

    # Configure and fit the Logistic Regression classifier
    logistic_classifier = LogisticRegression(class_weight='balanced', solver='lbfgs', multi_class='multinomial', max_iter=1000)
    
    # Perform cross-validated predictions
    predicted_labels = cross_val_predict(logistic_classifier, features, labels, cv=num_folds)

    # Print classification report comparing predictions with actual labels
    print(classification_report(labels, predicted_labels))
    
    # Generate and print a confusion matrix
    confusion_mat = confusion_matrix(labels, predicted_labels)
    print('\n')
    print(pd.crosstab(labels, predicted_labels, rownames=['Actual'], colnames=['Predicted'], margins=True))

if __name__ == '__main__':
    # Collect command line arguments, omitting the script name
    arguments = sys.argv[1:]
    if not arguments:
        print('usage: python run_sklearn_model_performance.py [path_to_feature_file]')
        sys.exit(1)
    data_file_path = arguments[0]
    process_data(data_file_path)