import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Any
from model_training import train_and_evaluate, get_models
from visualization import plot_metrics_comparison
from explanations import explain_with_lime

def run_bag_of_words(train_reviews: List[str], train_sentiments: List[str], 
                     test_reviews: List[str], test_sentiments: List[str], models: List[str]) -> None:
    """
    Train and evaluate models using Bag of Words representation.

    Args:
        train_reviews (List[str]): List of training reviews.
        train_sentiments (List[str]): List of training sentiments.
        test_reviews (List[str]): List of testing reviews.
        test_sentiments (List[str]): List of testing sentiments.
        models (List[str]): List of models to train and evaluate.
    """
    vectorizer = CountVectorizer()
    try:
        X_train = vectorizer.fit_transform(train_reviews)
        X_test = vectorizer.transform(test_reviews)
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return

    metrics = {
        'model': [],
        'accuracy': [],
        'recall': [],
        'precision': []
    }

    models_dict = get_models()
    valid_models = set(models_dict.keys()).union({'all'})

    if not all(model in valid_models for model in models):
        raise ValueError(f"Invalid model specified. Choose from {valid_models}")

    trained_models = {}
    if 'all' in models:
        for model_key, model in models_dict.items():
            success = train_and_evaluate(model, model_key, X_train, train_sentiments, X_test, test_sentiments, metrics, vectorizer)
            if success:
                trained_models[model_key] = model
    else:
        for model_key in models:
            if model_key in models_dict:
                model = models_dict[model_key]
                success = train_and_evaluate(model, model_key, X_train, train_sentiments, X_test, test_sentiments, metrics, vectorizer)
                if success:
                    trained_models[model_key] = model

    explain_with_lime(vectorizer, trained_models, test_reviews)
    plot_metrics_comparison(metrics)
