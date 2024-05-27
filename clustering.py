from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def run_clustering(train_reviews, train_sentiments, test_reviews, test_sentiments):
    # tf_idf
    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(train_reviews)
    X_test = tfidf_vectorizer.transform(test_reviews)
    
    # Scaling
    scaler = StandardScaler(with_mean=False)  # Set with_mean=False for sparse data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Perform K-means clustering
    k = 2  # Number of clusters (assuming binary sentiment classification like we have with positive/negative)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    
    # Assign sentiment labels to clusters
    train_cluster_labels = kmeans.labels_
    test_cluster_labels = kmeans.predict(X_test_scaled)
    
    # Map cluster labels to sentiment labels
    train_cluster_sentiments = []
    test_cluster_sentiments = []
    
    for i in range(k):
        train_cluster_data = train_sentiments[train_cluster_labels == i]
        if not train_cluster_data.empty:
            train_cluster_sentiments.append(train_cluster_data.mode()[0])
        else:
            train_cluster_sentiments.append(None)
        
        test_cluster_data = test_sentiments[test_cluster_labels == i]
        if not test_cluster_data.empty:
            test_cluster_sentiments.append(test_cluster_data.mode()[0])
        else:
            test_cluster_sentiments.append(None)
    
    train_cluster_labels_mapped = [train_cluster_sentiments[label] for label in train_cluster_labels]
    test_cluster_labels_mapped = [test_cluster_sentiments[label] for label in test_cluster_labels]
    
    # Evaluation
    train_accuracy = accuracy_score(train_sentiments, train_cluster_labels_mapped)
    test_accuracy = accuracy_score(test_sentiments, test_cluster_labels_mapped)
    print("Clustering Training Accuracy:", train_accuracy)
    print("Clustering Testing Accuracy:", test_accuracy)
    print(classification_report(test_sentiments, test_cluster_labels_mapped))