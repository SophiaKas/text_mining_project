import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Dict

def plot_metrics_comparison(metrics: Dict[str, List]) -> None:
    """
    Plot a comparison of model metrics.

    Args:
        metrics (Dict[str, List]): Dictionary containing model metrics.
    """
    models = metrics['model']
    accuracy = metrics['accuracy']
    recall = metrics['recall']
    precision = metrics['precision']
    
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='b')
    rects2 = ax.bar(x, recall, width, label='Recall', color='g')
    rects3 = ax.bar(x + width, precision, width, label='Precision', color='r')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Models by Accuracy, Recall and Precision')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    fig.tight_layout()
    plt.show()

def visualize_tsne(X_tsne: np.ndarray, labels: List[str], title: str) -> None:
    """
    Visualize the t-SNE representation of the feature space.

    Args:
        X_tsne (np.ndarray): The feature space matrix.
        labels (List[str]): The labels for the data points.
        title (str): The title for the plot.
    """
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_2d = tsne.fit_transform(X_tsne.toarray())
    
    # Convert labels to numeric values
    label_colors = [1 if label == 'positive' else 0 for label in labels]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=label_colors, cmap='viridis', label=labels)
    
    # Create a legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ['negative', 'positive'], title="Sentiment")
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.colorbar(label='Sentiment')
    plt.show()
