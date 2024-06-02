import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from typing import List, Dict, Any
from visualization import visualize_tsne

def get_models() -> Dict[str, Any]:
    return {
        'lr': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100),
        'svm': SVC(probability=True)
    }

def train_and_evaluate(model: Any, model_key: str, X_train: np.ndarray, train_sentiments: List[str], 
                       X_test: np.ndarray, test_sentiments: List[str], metrics: Dict[str, List], vectorizer: Any,
                       display_importance: bool = False, visualize_tsne_flag: bool = False) -> bool:
    """
    Train and evaluate a model, updating the metrics dictionary.

    Args:
        model (Any): The machine learning model.
        model_key (str): Key identifying the model.
        X_train (np.ndarray): Training feature matrix.
        train_sentiments (List[str]): Training labels.
        X_test (np.ndarray): Testing feature matrix.
        test_sentiments (List[str]): Testing labels.
        metrics (Dict[str, List]): Dictionary to store evaluation metrics.
        vectorizer (Any): Vectorizer used for feature extraction.
        display_importance (bool): Flag to display feature importance.
        visualize_tsne_flag (bool): Flag to visualize t-SNE.

    Returns:
        bool: True if the model was successfully trained, False otherwise.
    """
    try:
        model.fit(X_train, train_sentiments)
        predictions = model.predict(X_test)
    except Exception as e:
        print(f"Error training {model_key}: {e}")
        return False
    
    update_metrics(metrics, model_key, test_sentiments, predictions)
    print_classification_report(test_sentiments, predictions, model_key)
    
    if display_importance:
        display_feature_importance(model, vectorizer, top_n=20)
    if visualize_tsne_flag:
        visualize_tsne(X_train, train_sentiments, f't-SNE Visualization of Feature Space in {model_key}')
    
    return True

def update_metrics(metrics: Dict[str, List], model_key: str, test_sentiments: List[str], predictions: List[str]) -> None:
    """
    Update the metrics dictionary with the performance of a model.

    Args:
        metrics (Dict[str, List]): Dictionary to store evaluation metrics.
        model_key (str): Key identifying the model.
        test_sentiments (List[str]): Ground truth labels.
        predictions (List[str]): Predicted labels.
    """
    metrics['model'].append(model_key)
    metrics['accuracy'].append(accuracy_score(test_sentiments, predictions))
    metrics['recall'].append(recall_score(test_sentiments, predictions, average='weighted'))
    metrics['precision'].append(precision_score(test_sentiments, predictions, average='weighted'))

def print_classification_report(test_sentiments: List[str], predictions: List[str], model_key: str) -> None:
    """
    Print the classification report for a model.

    Args:
        test_sentiments (List[str]): Ground truth labels.
        predictions (List[str]): Predicted labels.
        model_key (str): Key identifying the model.
    """
    print(f"Classification Report for {model_key}:\n")
    report = classification_report(test_sentiments, predictions, target_names=['negative', 'positive'])
    print(report)

def display_feature_importance(model: Any, vectorizer: Any, top_n: int = 10) -> None:
    """
    Display the top N feature importances for a model.

    Args:
        model (Any): The machine learning model.
        vectorizer (Any): Vectorizer used for feature extraction.
        top_n (int): Number of top features to display.
    """
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        indices = np.argsort(np.abs(coefficients))[::-1]
        print_feature_importance(feature_names, coefficients, indices, top_n)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print_feature_importance(feature_names, importances, indices, top_n)

def print_feature_importance(feature_names: List[str], importances: np.ndarray, indices: np.ndarray, top_n: int) -> None:
    """
    Print the feature importances.

    Args:
        feature_names (List[str]): List of feature names.
        importances (np.ndarray): Array of feature importances.
        indices (np.ndarray): Indices of the top features.
        top_n (int): Number of top features to display.
    """
    for i in range(min(top_n, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
