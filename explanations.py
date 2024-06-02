import lime
import lime.lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def explain_with_lime(vectorizer, models_dict, test_reviews, num_features=10, num_samples=1000):
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    
    for model_key, model in models_dict.items():
        display(HTML(f"<h2>Explaining model: {model_key}</h2>"))
        pipeline = make_pipeline(vectorizer, model)
        try:
            check_is_fitted(model)
            predictions = model.predict_proba(vectorizer.transform(test_reviews))
            
            # Choosing the review with the prediction closest to 0.5 for a mixed sentiment
            mixed_index = np.argmin(np.abs(predictions[:, 1] - 0.5))
            review_mixed = test_reviews[mixed_index]
            
            exp_mixed = explainer.explain_instance(review_mixed, pipeline.predict_proba, num_features=num_features, num_samples=num_samples)
            display(HTML(f"<h3>Mixed Sentiment Review</h3>"))
            display(HTML(exp_mixed.as_html()))
            
            # Choosing the review with the most positive sentiment
            positive_index = np.argmax(predictions[:, 1])
            review_positive = test_reviews[positive_index]
            
            exp_positive = explainer.explain_instance(review_positive, pipeline.predict_proba, num_features=num_features, num_samples=num_samples)
            display(HTML(f"<h3>Positive Sentiment Review</h3>"))
            display(HTML(exp_positive.as_html()))
            
            # Choosing the review with the most negative sentiment
            negative_index = np.argmin(predictions[:, 1])
            review_negative = test_reviews[negative_index]
            
            exp_negative = explainer.explain_instance(review_negative, pipeline.predict_proba, num_features=num_features, num_samples=num_samples)
            display(HTML(f"<h3>Negative Sentiment Review</h3>"))
            display(HTML(exp_negative.as_html()))
            
        except Exception as e:
            display(HTML(f"<p style='color:red;'>Model {model_key} encountered an error: {str(e)}. Skipping explanation.</p>"))
