#!/usr/bin/env python
# coding: utf-8

# In[1]:


# open python and nltk packages needed for processing
import sentiment_read_subjectivity
import os
import sys
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
import re

stopwords = nltk.corpus.stopwords.words('english')
new_stopwords = [word for word in stopwords if word not in ['not', 'no', 'can', 'don', 't']]


def preprocess_document(document):
  # "Preprocessing documents"  
  # "create list of lower case words"
  words_list = re.split('\s+', document.lower())

  punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
  words_list = [punctuation.sub("", word) for word in words_list] 
  final_words_list = []
  for word in words_list:
    if word not in new_stopwords:
      final_words_list.append(word)
  line = " ".join(final_words_list)
  return line 

def get_words_from_docs(docs):
  all_words = []
  for (words, sentiment) in docs:
    # more than 3 length
    possible_words = [x for x in words if len(x) >= 3]
    all_words.extend(possible_words)
  return all_words

def get_words_from_docs_usual(docs):
  all_words = []
  for (words, sentiment) in docs:
    all_words.extend(words)
  return all_words  

# get all words from tokens
def get_words_from_test_dataset(lines):
  all_words = []
  for id, words in lines:
    all_words.extend(words)
  return all_words


def write_feature_sets(feature_sets, output_path):
    # open output_path for writing
    file = open(output_path, 'w')
    # get the feature names from the feature dictionary in the first feature_set
    feature_names = feature_sets[0][0].keys()
    # create the first line of the file as comma-separated feature names
    #    with the word class as the last feature name
    feature_names_line = ''
    for feature_name in feature_names:
        # replace forbidden characters with text abbreviations
        feature_name = feature_name.replace(',','CM')
        feature_name = feature_name.replace("'","DQ")
        feature_name = feature_name.replace('"','QU')
        feature_names_line += feature_name + ','
    feature_names_line += 'class'
    # write this as the first line in the csv file
    file.write(feature_names_line)
    file.write('\n')
    for feature_set in feature_sets:
        feature_line = ''
        for key in feature_names:
          feature_line += str(feature_set[0][key]) + ','
        if feature_set[1] == 0:
          feature_line += str("neg")
        elif feature_set[1] == 1:
          feature_line += str("sneg")
        elif feature_set[1] == 2:
          feature_line += str("neu")
        elif feature_set[1] == 3:
          feature_line += str("spos")
        elif feature_set[1] == 4:
          feature_line += str("pos")
        # write each feature set values to the file
        file.write(feature_line)
        file.write('\n')
    file.close()


def get_word_features(word_list):
  word_list = nltk.FreqDist(word_list)
  word_features = [w for (w, c) in word_list.most_common(200)] 
  return word_features    


def usual_features(document, word_features):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  return features


def bigram_document_features(document, word_features, bigram_features):
  document_words = set(document)
  document_bigrams = nltk.bigrams(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  for bigram in bigram_features:
    features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
  return features

def get_bigram_features(tokens):
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(tokens, window_size=3)
  bigram_features = finder.nbest(bigram_measures.chi_sq, 3000)
  return bigram_features[:500]

def trigram_document_features(document, word_features, trigram_features):
  document_words = set(document)
  document_trigrams = nltk.trigrams(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  for trigram in trigram_features:
    #print(trigram)
    features['trigram({} {} {})'.format(trigram[0], trigram[1], trigram[2])] = (trigram in document_trigrams)    
  return features

def get_trigram_features(tokens):
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  finder = TrigramCollocationFinder.from_words(tokens, window_size=3)
  #finder.apply_freq_filter(6)
  trigram_features = finder.nbest(trigram_measures.chi_sq, 3000)
  return trigram_features[:500]

def read_subjectivity(path):
  flexicon = open(path, 'r')
  # initialize an empty dictionary
  sl_dict = { }
  for line in flexicon:
    fields = line.split()
    strength = fields[0].split("=")[1]
    word = fields[2].split("=")[1]
    pos_tag = fields[3].split("=")[1]
    stemmed = fields[4].split("=")[1]
    polarity = fields[5].split("=")[1]
    if (stemmed == 'y'):
      is_stemmed = True
    else:
      is_stemmed = False
    # put a dictionary entry with the word as the keyword
    #     and a list of the other values
    sl_dict[word] = [strength, pos_tag, is_stemmed, polarity]
  return sl_dict

SL_path = "./SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
SL = read_subjectivity(SL_path)
def SL_features(document, word_features, SL):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  # count variables for the 4 classes of subjectivity
  weak_pos = 0
  strong_pos = 0
  weak_neg = 0
  strong_neg = 0
  for word in document_words:
    if word in SL:
      strength, pos_tag, is_stemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weak_pos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strong_pos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weak_neg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strong_neg += 1
      features['positivecount'] = weak_pos + (2 * strong_pos)
      features['negativecount'] = weak_neg + (2 * strong_neg)
  
  if 'positivecount' not in features:
    features['positivecount'] = 0
  if 'negativecount' not in features:
    features['negativecount'] = 0      
  return features


negation_words = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather',
                 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

def NOT_features(document, word_features, negation_words):
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = False
    features['contains(NOT{})'.format(word)] = False
  # go through document words in order
  for i in range(0, len(document)):
    word = document[i]
    if ((i + 1) < len(document)) and (word in negation_words):
      i += 1
      features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
    else:
      if ((i + 3) < len(document)) and (word.endswith('n') and document[i+1] == "'" and document[i+2] == 't'):
        i += 3
        features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
      else:
        features['contains({})'.format(word)] = (word in word_features)
  return features


def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    num_noun = 0
    num_verb = 0
    num_adj = 0
    num_adverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): num_noun += 1
        if tag.startswith('V'): num_verb += 1
        if tag.startswith('J'): num_adj += 1
        if tag.startswith('R'): num_adverb += 1
    features['nouns'] = num_noun
    features['verbs'] = num_verb
    features['adjectives'] = num_adj
    features['adverbs'] = num_adverb
    return features



def process_kaggle(dir_path, limit_str, seed=82):
  # convert the limit argument from a string to an int
  limit = int(limit_str)

  random.seed(seed)
  os.chdir(dir_path)
  
  file = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrase_data = []
  for line in file:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrase_data.append(line.split('\t')[2:4])

  random.shuffle(phrase_data)
  phrase_list = phrase_data[:limit]

  print('Read', len(phrase_data), 'phrases, using', len(phrase_list), 'random phrases')
  
  # create list of phrase documents as (list of words, label)
  phrase_docs = []
  phrase_docs_without = []
  # add all the phrases
  for phrase in phrase_list:

    # without preprocessing
    tokens = nltk.word_tokenize(phrase[0])
    phrase_docs_without.append((tokens, int(phrase[1])))
    
    # with preprocessing
    tokenizer = RegexpTokenizer(r'\w+')
    phrase[0] = preprocess_document(phrase[0])
    tokens = tokenizer.tokenize(phrase[0])
    phrase_docs.append((tokens, int(phrase[1])))
  
  # possibly filter tokens
  normal_tokens = get_words_from_docs_usual(phrase_docs_without)
  preprocessed_tokens = get_words_from_docs(phrase_docs)


  word_features = get_word_features(normal_tokens)
  feature_sets_without_preprocessing = [(usual_features(d, word_features), s) for (d, s) in phrase_docs_without]
  write_feature_sets(feature_sets_without_preprocessing, "feature_sets_without_preprocessing.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with normal features, without preprocessing steps: ")
  accuracy_calculation(feature_sets_without_preprocessing)


  word_features = get_word_features(preprocessed_tokens)

  feature_sets = [(usual_features(d, word_features), s) for (d, s) in phrase_docs]
  write_feature_sets(feature_sets, "feature_sets.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with preprocessed features: ")
  accuracy_calculation(feature_sets)
  
  SL_feature_sets = [(SL_features(d, word_features, SL), c) for (d, c) in phrase_docs]
  write_feature_sets(SL_feature_sets, "features_SL.csv")
  #print SL_feature_sets[0]
  print ("---------------------------------------------------")
  print ("Accuracy with SL_feature_sets: ")
  accuracy_calculation(SL_feature_sets)

  NOT_feature_sets = [(NOT_features(d, word_features, negation_words), c) for (d, c) in phrase_docs]
  #print NOT_feature_sets[0]
  write_feature_sets(SL_feature_sets, "features_NOT.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with NOT_feature_sets: ")
  accuracy_calculation(NOT_feature_sets)

  POS_feature_sets = [(POS_features(d, word_features), c) for (d, c) in phrase_docs]
  #print NOT_feature_sets[0]
  write_feature_sets(POS_feature_sets, "features_POS.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with POS_feature_sets: ")
  accuracy_calculation(POS_feature_sets)

  bigram_features = get_bigram_features(preprocessed_tokens)
  #print(bigram_features[0])
  bigram_feature_sets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in phrase_docs]
  #print(bigram_feature_sets[0])
  write_feature_sets(bigram_feature_sets, "features_bigram.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with bigram feature sets: ")
  accuracy_calculation(bigram_feature_sets)

  trigram_features = get_trigram_features(preprocessed_tokens)
  #print (trigram_features[0])
  trigram_feature_sets = [(trigram_document_features(d, word_features, trigram_features), c) for (d, c) in phrase_docs]
  #print (trigram_feature_sets[0])
  write_feature_sets(trigram_feature_sets, "features_trigram.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with Trigram feature sets: ")
  accuracy_calculation(trigram_feature_sets)
  
  combined_document_features = get_combined_document_features(preprocessed_tokens)
  #print (combined_document_features[0])
  features_combined_sets= [combined_document_features(doc, word_features, SL_features, bigram_features) for doc, _ in phrase_docs]
   #print (features_combined_sets[0])
  write_feature_sets(features_combined_sets, "features_combined.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with Combined feature sets: ")
  accuracy_calculation(features_combined_sets)

  

def accuracy_calculation(feature_sets):
    print("Checking feature sets for None types...")
    for features, label in feature_sets:
        if features is None:
            print("Found None in feature sets, which should not happen.")
            continue  # Or raise an error to handle this case

    print("Training and testing a classifier")
    training_size = int(0.1 * len(feature_sets))
    test_set = feature_sets[:training_size]
    training_set = feature_sets[training_size:]
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Accuracy of classifier:", nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features()


def print_confusion_matrix(classifier_type, test_set):
  ref_list = []
  test_list = []
  for (features, label) in test_set:
    ref_list.append(label)
    test_list.append(classifier_type.classify(features))
  
  print (" ")
  print ("The confusion matrix")
  cm = ConfusionMatrix(ref_list, test_list)
  print (cm)

def create_test_submission(feature_sets, test_feature_sets, file_name):
  print ("---------------------------------------------------")
  print ("Training and testing a classifier ")
  test_set = test_feature_sets
  training_set = feature_sets
  classifier = nltk.NaiveBayesClassifier.train(training_set)
  file_writer = open(file_name, "w")
  file_writer.write("PhraseId" + ',' + "Sentiment" + '\n')
  for test, id in test_feature_sets:
    file_writer.write(str(id) + ',' + str(classifier.classify(test)) + '\n')
  file_writer.close()


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


def combined_document_features(document, word_features, SL_featuresets, bigram_featuresets):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}

    print("Processing document words:", document_words)  # Debug print

    if not document_words:
        print("Empty document_words detected.")  # Debug print
        return {}

    # The following segment should be re-examined to ensure it doesn't cause any unexpected results
    pos_word, neut_word, neg_word = 0, 0, 0
    for word in document_words:
        if word in SL_featuresets[0]:  # SL_featuresets[0] might be incorrect if SL_featuresets is not structured as expected
            pos_word += 1
        if word in SL_featuresets[1]:
            neut_word += 1
        if word in SL_featuresets[2]:
            neg_word += 1
        features['positivecount'] = pos_word
        features['neutralcount'] = neut_word
        features['negativecount'] = neg_word

    # Ensure bigrams and other features are added correctly
    for word in word_features:
        features['V_{}'.format(word)] = word in document_words
        features['V_NOT{}'.format(word)] = False  # Adjust logic here if necessary

    for bigram in bigram_featuresets:  # Ensure bigram_featuresets is correctly a list of bigrams
        features['B_{}_{}'.format(bigram[0], bigram[1])] = bigram in document_bigrams

    print("Features created:", features)  # Debug print
    return features
   



if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    process_kaggle(sys.argv[1], sys.argv[2])

