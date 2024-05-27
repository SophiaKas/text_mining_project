from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_bag_of_words(train_reviews, train_sentiments, test_reviews, test_sentiments, models):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_reviews)
    X_test = vectorizer.transform(test_reviews)
    
    if 'lr' in models or 'all' in models:
        # Logistic Regression model
        lr_model = LogisticRegression(max_iter=1000) #tried 1000 iteration since there are usually warnings if I dont do this
        lr_model.fit(X_train, train_sentiments)
        lr_predictions = lr_model.predict(X_test)
        print("Logistic Regression (Bag of Words) Accuracy:", accuracy_score(test_sentiments, lr_predictions))
    
    if 'rf' in models or 'all' in models:
        # Random Forest model
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, train_sentiments)
        rf_predictions = rf_model.predict(X_test)
        print("Random Forest (Bag of Words) Accuracy:", accuracy_score(test_sentiments, rf_predictions))
    
    if 'svm' in models or 'all' in models:
        # SVM model
        svm_model = SVC()
        svm_model.fit(X_train, train_sentiments)
        svm_predictions = svm_model.predict(X_test)
        print("SVM (Bag of Words) Accuracy:", accuracy_score(test_sentiments, svm_predictions))