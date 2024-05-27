from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text

def run_topic_modeling(train_data, test_data):
    # Preprocessing
    train_data['preprocessed_review'] = train_data['review'].apply(preprocess_text)
    test_data['preprocessed_review'] = test_data['review'].apply(preprocess_text)
    
    # Create a CountVectorizer to convert reviews into a matrix of word counts
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X_train = vectorizer.fit_transform(train_data['preprocessed_review'])
    X_test = vectorizer.transform(test_data['preprocessed_review'])
    
    # Apply LDA topic modeling
    num_topics = 5  # Specify the number of topics you want to discover
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X_train)
    
    # Print topic top words
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-10:-1]]))
    
    # Train a sentiment classification model (Random Forest)
    y_train = train_data['sentiment']
    y_test = test_data['sentiment']
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # evaluation
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSentiment Classification Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))