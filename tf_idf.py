from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_tf_idf(train_reviews, train_sentiments, test_reviews, test_sentiments, models):
    # tf-idf
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_reviews)
    X_test_tfidf = tfidf_vectorizer.transform(test_reviews)
    
    if 'lr' in models or 'all' in models:
        # Logistic Regression model for tf-idf
        lr_model_tfidf = LogisticRegression(max_iter=1000)
        lr_model_tfidf.fit(X_train_tfidf, train_sentiments)
        lr_predictions_tfidf = lr_model_tfidf.predict(X_test_tfidf)
        print("Logistic Regression (tf-idf) Accuracy:", accuracy_score(test_sentiments, lr_predictions_tfidf))
    
    if 'rf' in models or 'all' in models:
        # Random Forest model for tf-idf
        rf_model_tfidf = RandomForestClassifier()
        rf_model_tfidf.fit(X_train_tfidf, train_sentiments)
        rf_predictions_tfidf = rf_model_tfidf.predict(X_test_tfidf)
        print("Random Forest (tf-idf) Accuracy:", accuracy_score(test_sentiments, rf_predictions_tfidf))
    
    if 'svm' in models or 'all' in models:
        # SVM model for tf-idf
        svm_model_tfidf = SVC()
        svm_model_tfidf.fit(X_train_tfidf, train_sentiments)
        svm_predictions_tfidf = svm_model_tfidf.predict(X_test_tfidf)
        print("SVM (tf-idf) Accuracy:", accuracy_score(test_sentiments, svm_predictions_tfidf))