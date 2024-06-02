from sklearn.feature_extraction.text import TfidfVectorizer
from model_training import train_and_evaluate, get_models
from visualization import plot_metrics_comparison
from explanations import explain_with_lime
from typing import List, Dict

def run_tf_idf(train_reviews: List[str], train_sentiments: List[str], 
               test_reviews: List[str], test_sentiments: List[str], models: List[str]) -> None:
    vectorizer = TfidfVectorizer()
    """
    Train and evaluate models using tf_idf representation.

    Args:
        train_reviews (List[str]): List of training reviews.
        train_sentiments (List[str]): List of training sentiments.
        test_reviews (List[str]): List of testing reviews.
        test_sentiments (List[str]): List of testing sentiments.
        models (List[str]): List of models to train and evaluate.
    """
    X_train = vectorizer.fit_transform(train_reviews)
    X_test = vectorizer.transform(test_reviews)

    metrics = {
        'model': [],
        'accuracy': [],
        'recall': [],
        'precision': []
    }

    models_dict = get_models()

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
