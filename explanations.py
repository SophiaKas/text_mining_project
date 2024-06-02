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

def save_html(html, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(html)

def plot_probabilities(predictions, title):
    plt.figure(figsize=(10, 6))
    plt.hist(predictions[:, 1], bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Predicted Probability of Positive Sentiment')
    plt.ylabel('Number of Reviews')
    plt.show()

def explain_with_lime(vectorizer, models_dict, test_reviews, num_features=7, num_samples=15000):
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    
    for model_key, model in models_dict.items():
        display(HTML(f"<h2>Explaining model: {model_key}</h2>"))
        pipeline = make_pipeline(vectorizer, model)
        try:
            check_is_fitted(model)
            predictions = model.predict_proba(vectorizer.transform(test_reviews))
            
            plot_probabilities(predictions, f'Distribution of Predicted Probabilities for {model_key}')
            
            # Reviews with confidence just above and below 0.5
            threshold = 0.05
            near_50_indices = np.where((predictions[:, 1] > 0.5 - threshold) & (predictions[:, 1] < 0.5 + threshold))[0]
            display(HTML(f"<h3>Reviews with Confidence Near 50%</h3>"))
            for idx in near_50_indices[:5]:
                review_near_50 = test_reviews[idx]
                exp_near_50 = explainer.explain_instance(review_near_50, pipeline.predict_proba, num_features=num_features, num_samples=num_samples)
                html_near_50 = exp_near_50.as_html()
                display(HTML(html_near_50))
                save_html(html_near_50, f"{model_key}_near_50_{idx}.html")
            
            # Reviews with high positive sentiment confidence
            high_pos_indices = np.where(predictions[:, 1] > 0.9)[0]
            display(HTML(f"<h3>High Confidence Positive Sentiment Reviews</h3>"))
            for idx in high_pos_indices[:5]:
                review_high_pos = test_reviews[idx]
                exp_high_pos = explainer.explain_instance(review_high_pos, pipeline.predict_proba, num_features=num_features, num_samples=num_samples)
                html_high_pos = exp_high_pos.as_html()
                display(HTML(html_high_pos))
                save_html(html_high_pos, f"{model_key}_high_pos_{idx}.html")
            
            # Reviews with high negative sentiment confidence
            high_neg_indices = np.where(predictions[:, 1] < 0.1)[0]
            display(HTML(f"<h3>High Confidence Negative Sentiment Reviews</h3>"))
            for idx in high_neg_indices[:5]:
                review_high_neg = test_reviews[idx]
                exp_high_neg = explainer.explain_instance(review_high_neg, pipeline.predict_proba, num_features=num_features, num_samples=num_samples)
                html_high_neg = exp_high_neg.as_html()
                display(HTML(html_high_neg))
                save_html(html_high_neg, f"{model_key}_high_neg_{idx}.html")
            
        except Exception as e:
            display(HTML(f"<p style='color:red;'>Model {model_key} encountered an error: {str(e)}. Skipping explanation.</p>"))
