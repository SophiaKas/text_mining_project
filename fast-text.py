import fasttext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,confusion_matrix
import seaborn as sns

# Define the plot_metrics function
def plot_metrics(accuracies, precisions, recalls, model_names, confusion_matrices):
    num_models = len(model_names)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))  # Increase ncols to accommodate the confusion matrix
    
    # Plot accuracies
    axes[0].barh(model_names, accuracies, color='skyblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlim(0.8, 1.0)  # Set x-axis limit from 0.8 to 1.0
    
    # Plot precisions
    axes[1].barh(model_names, precisions, color='salmon')
    axes[1].set_xlabel('Precision')
    axes[1].set_title('Model Precision')
    
    # Plot recalls
    axes[2].barh(model_names, recalls, color='lightgreen')
    axes[2].set_xlabel('Recall')
    axes[2].set_title('Model Recall')
    
    # Plot confusion matrices
    for i, (name, matrix) in enumerate(zip(model_names, confusion_matrices)):
        sns.heatmap(matrix, annot=True, cmap="Blues", ax=axes[i+3])  # Adjust ax index
        axes[i+3].set_title(f'Confusion Matrix - {name}')
        axes[i+3].set_xlabel('Predicted')
        axes[i+3].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()




# Load data
data = pd.read_csv('./IMDB Dataset reduced.csv')

# Load Spacy model
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation

# Tokenization and preprocessing function
def spacy_tokenizer(sentence):
    doc = nlp(sentence)
    mytokens = [word.lemma_.lower().strip() for word in doc]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return " ".join(mytokens)

# Preprocess data
data['review'] = data['review'].apply(spacy_tokenizer)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, stratify=data['sentiment'])

# Convert data to FastText format
train_file = 'train.txt'
test_file = 'test.txt'

with open(train_file, 'w') as f:
    for text, label in zip(X_train, y_train):
        f.write(f'__label__{label} {text}\n')

with open(test_file, 'w') as f:
    for text, label in zip(X_test, y_test):
        f.write(f'__label__{label} {text}\n')

# Train FastText classifier
model = fasttext.train_supervised(input=train_file)

# Test the classifier
result = model.test(test_file)

# Print evaluation results
print("Precision:", result[1])
print("Recall:", result[2])
print("Number of examples:", result[0])

# Predict labels for new texts
def predict_label(text):
    label = model.predict(text)
    return label[0][0]

X_test_predicted = X_test.apply(predict_label)
y_test_predicted = [int(label.split('__label__')[1]) for label in X_test_predicted]

# Calculate metrics
accuracy = np.mean(y_test_predicted == y_test)
precision = precision_score(y_test, y_test_predicted, pos_label=1)
recall = recall_score(y_test, y_test_predicted, pos_label=1)


# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Plot metrics
plot_metrics([accuracy], [precision], [recall], ['FastText'], [confusion_matrix])