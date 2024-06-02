from sklearn.feature_extraction.text import CountVectorizer
from typing import List

def vectorize_data(train_reviews: List[str], test_reviews: List[str]):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_reviews)
    X_test = vectorizer.transform(test_reviews)
    return X_train, X_test, vectorizer
