import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_text
from bag_of_words import run_bag_of_words
from tf_idf import run_tf_idf
from topic_modeling import run_topic_modeling
from clustering import run_clustering
import os

default_file_path = './IMDB Dataset.csv'

# Check if the file is in the same folder
if os.path.exists(default_file_path):
    imdb_data = pd.read_csv(default_file_path)
else:
    user_file_path = input("File not found. Provide new path: ")
    
    if os.path.exists(user_file_path):
        imdb_data = pd.read_csv(user_file_path)
    else:
        print("Invalid file path")
        exit()

# Ask which version to run
print("Choose the version to run:")
print("1. Bag of words")
print("2. Tf-idf")
print("3. Topic Modeling")
print("4. Clustering")
version = input("Enter number: ")

if version not in {'1', '2', '3', '4'}:
    print("Invalid option. Choose 1, 2, 3, or 4.")
    exit()

# Prompt user to choose sampling size
print("Choose the sampling method:")
print("1. Custom sample size")
print("2. Use 0.8/0.2 split")
sampling_method = input("Enter the sampling method number: ")

if sampling_method == "1":
    sample_size = int(input("Enter the sample size: "))

    if sample_size < len(imdb_data):
        imdb_data = imdb_data.sample(n=sample_size, random_state=None)
    
    # Split the data into training and testing sets
    train_data = imdb_data.sample(frac=0.8, random_state=None)
    test_data = imdb_data.drop(train_data.index)
else:  # 0.8 / 0.2 split
    train_data, test_data = train_test_split(imdb_data, test_size=0.2, random_state=None)

# Extraction of reviews and sentiments
train_reviews = train_data.review
train_sentiments = train_data.sentiment
test_reviews = test_data.review
test_sentiments = test_data.sentiment

# Preprocessing
train_reviews = train_reviews.apply(preprocess_text)
test_reviews = test_reviews.apply(preprocess_text)

if version == "1":
    print("Choose the learning models to run ('lr,rf,svm' or 'all'):")
    models = input("Enter the model: ").split(',')
    run_bag_of_words(train_reviews.tolist(), train_sentiments.tolist(), test_reviews.tolist(), test_sentiments.tolist(), models)  # converting everything to lists
elif version == "2":
    print("Choose the learning models to run ('lr,rf,svm' or 'all'):")
    models = input("Enter the model: ").split(',')
    run_tf_idf(train_reviews.tolist(), train_sentiments.tolist(), test_reviews.tolist(), test_sentiments.tolist(), models)  # converting everything to lists
elif version == "3":
    run_topic_modeling(train_data, test_data)
elif version == "4":
    run_clustering(train_reviews.tolist(), train_sentiments.tolist(), test_reviews.tolist(), test_sentiments.tolist())  # converting everything to lists
else:
    print("Choose 1, 2, 3, or 4.")
