import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_tsne(X, labels, title):
    # Convert labels to numeric values
    label_map = {'negative': 0, 'positive': 1}
    labels_numeric = labels.map(label_map)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_numeric, cmap='viridis')
    plt.colorbar(label='Sentiment')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()