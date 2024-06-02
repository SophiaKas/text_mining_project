# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from typing import List, Dict, Any
# from model_training import train_and_evaluate, get_models
# from visualization import plot_metrics_comparison
# from explanations import explain_with_lime

# def run_bag_of_words(train_reviews: List[str], train_sentiments: List[str], 
#                      test_reviews: List[str], test_sentiments: List[str], models: List[str]) -> None:
#     """
#     Train and evaluate models using Bag of Words representation.

#     Args:
#         train_reviews (List[str]): List of training reviews.
#         train_sentiments (List[str]): List of training sentiments.
#         test_reviews (List[str]): List of testing reviews.
#         test_sentiments (List[str]): List of testing sentiments.
#         models (List[str]): List of models to train and evaluate.
#     """
#     vectorizer = CountVectorizer()
#     try:
#         X_train = vectorizer.fit_transform(train_reviews)
#         X_test = vectorizer.transform(test_reviews)
#     except Exception as e:
#         print(f"Error during vectorization: {e}")
#         return

#     metrics = {
#         'model': [],
#         'accuracy': [],
#         'recall': [],
#         'precision': []
#     }

#     models_dict = get_models()
#     valid_models = set(models_dict.keys()).union({'all'})

#     if not all(model in valid_models for model in models):
#         raise ValueError(f"Invalid model specified. Choose from {valid_models}")

#     trained_models = {}
#     if 'all' in models:
#         for model_key, model in models_dict.items():
#             success = train_and_evaluate(model, model_key, X_train, train_sentiments, X_test, test_sentiments, metrics, vectorizer)
#             if success:
#                 trained_models[model_key] = model
#     else:
#         for model_key in models:
#             if model_key in models_dict:
#                 model = models_dict[model_key]
#                 success = train_and_evaluate(model, model_key, X_train, train_sentiments, X_test, test_sentiments, metrics, vectorizer)
#                 if success:
#                     trained_models[model_key] = model

#     explain_with_lime(vectorizer, trained_models, test_reviews)
#     plot_metrics_comparison(metrics)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def run_bag_of_words(train_reviews, train_sentiments, test_reviews, test_sentiments, models):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_reviews)
    X_test = vectorizer.transform(test_reviews)
    
    metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    
    if 'lr' in models or 'all' in models:
        # Logistic Regression model
        lr_model = LogisticRegression(max_iter=1000)  # tried 1000 iteration since there are usually warnings if I dont do this
        lr_model.fit(X_train, train_sentiments)
        lr_predictions = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(test_sentiments, lr_predictions)
        lr_precision = precision_score(test_sentiments, lr_predictions, pos_label='positive')
        lr_recall = recall_score(test_sentiments, lr_predictions, pos_label='positive')
        metrics['Model'].append('Logistic Regression')
        metrics['Accuracy'].append(lr_accuracy)
        metrics['Precision'].append(lr_precision)
        metrics['Recall'].append(lr_recall)
        print("Logistic Regression (Bag of Words) Accuracy:", lr_accuracy)
        print("Precision: ", lr_precision)
        print("Recall :", lr_recall)
    
    if 'rf' in models or 'all' in models:
        # Random Forest model
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, train_sentiments)
        rf_predictions = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(test_sentiments, rf_predictions)
        rf_precision = precision_score(test_sentiments, rf_predictions, pos_label='positive')
        rf_recall = recall_score(test_sentiments, rf_predictions, pos_label='positive')
        metrics['Model'].append('Random Forest')
        metrics['Accuracy'].append(rf_accuracy)
        metrics['Precision'].append(rf_precision)
        metrics['Recall'].append(rf_recall)
        print("Random Forest (Bag of Words) Accuracy:", rf_accuracy)
        print("Precision: ", rf_precision)
        print("Recall :", rf_recall)
    
    if 'svm' in models or 'all' in models:
        # SVM model
        svm_model = SVC()
        svm_model.fit(X_train, train_sentiments)
        svm_predictions = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(test_sentiments, svm_predictions)
        svm_precision = precision_score(test_sentiments, svm_predictions, pos_label='positive')
        svm_recall = recall_score(test_sentiments, svm_predictions, pos_label='positive')
        metrics['Model'].append('SVM')
        metrics['Accuracy'].append(svm_accuracy)
        metrics['Precision'].append(svm_precision)
        metrics['Recall'].append(svm_recall)
        print("SVM (Bag of Words) Accuracy:", svm_accuracy)
        print("Precision: ", svm_precision)
        print("Recall :", svm_recall)
    
    # Plotting the metrics
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Horizontal bar plot for Accuracy
    ax[0].barh(metrics['Model'], metrics['Accuracy'], color='skyblue')
    ax[0].set_title('Accuracy:')
    ax[0].set_xlabel('Accuracy')

    # Horizontal bar plot for Precision
    ax[1].barh(metrics['Model'], metrics['Precision'], color='coral')
    ax[1].set_title('Precision:')
    ax[1].set_xlabel('Precision')

    # Horizontal bar plot for Recall
    ax[2].barh(metrics['Model'], metrics['Recall'], color='lightgreen')
    ax[2].set_title('Recall:')
    ax[2].set_xlabel('Recall')

    plt.tight_layout()

    # Save the plots
    fig.savefig('bow_model_metrics_comparison.png')
