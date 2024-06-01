import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import string
import gensim.downloader as api
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import seaborn as sns

count = 0

# Define the plot_metrics function
def plot_metrics(accuracies, precisions, recalls, model_names):
    num_models = len(model_names)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))  # Adjusted for the additional confusion matrix plot
    
    # Plot accuracies
    axes[0].barh(model_names, accuracies, color='skyblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlim(0.7, 1.0)  # Adjusted limits for accuracy
    
    # Plot precisions
    axes[1].barh(model_names, precisions, color='salmon')
    axes[1].set_xlabel('Precision')
    axes[1].set_title('Model Precision')
    axes[1].set_xlim(0.7, 1.0)  # Adjusted limits for precision
    
    # Plot recalls
    axes[2].barh(model_names, recalls, color='lightgreen')
    axes[2].set_xlabel('Recall')
    axes[2].set_title('Model Recall')
    axes[2].set_xlim(0.7, 1.0)  # Adjusted limits for recall
    
    # Calling the cf function
    best_model_index = accuracies.index(max(accuracies))
    best_model_name = model_names[best_model_index]
    best_model = models[best_model_name]
    best_predicted = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_predicted)
    plot_best_model_confusion_matrix(cm, best_model_name)
    
    plt.tight_layout()
    plt.show()

def plot_best_model_confusion_matrix(cm, best_model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{best_model_name} Confusion Matrix')
    plt.show()



print(list(gensim.downloader.info()['models'].keys()))
wv = api.load('word2vec-google-news-300')
print("loading done")


# Load data
data = pd.read_csv('./IMDB Dataset.csv')

# Load Spacy model
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation

# Tokenization and vectorization functions
def sent_vec(sent):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res/ctr
    return wv_res

def spacy_tokenizer(sentence):
    global count
    count = count + 1
    if count % 1000 == 0:
      print(count)
   
    doc = nlp(sentence)
    mytokens = [word.lemma_.lower().strip() for word in doc]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

# Tokenize and vectorize the data
data['tokens'] = data['review'].apply(spacy_tokenizer)
data['vec'] = data['tokens'].apply(sent_vec)

# Prepare features and target
X = data['vec'].to_list()
y = data['sentiment'].to_list()

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Train and evaluate models
accuracies = []
precisions = []
recalls = []
model_names = []

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    predicted = model.predict(X_test)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted, pos_label='positive')
    recall = recall_score(y_test, predicted, pos_label='positive')
    
    # Append metrics to lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    model_names.append(name)
    
    # Print metrics
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Precision: {precision}")
    print(f"{name} Recall: {recall}")
    print()

# Plot metrics
plot_metrics(accuracies, precisions, recalls, model_names)