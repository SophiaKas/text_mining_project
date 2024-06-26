# Text Mining

Based on: [Sentiment Analysis of IMDB Movie Reviews](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews)

Created for a group assignment as part of a Machine learning and Data mining course

## Usage

- Run `pip install -r requirements.txt` to install all requirements. Some nltk parts might need to be installed manually; check the terminal.
- Run `main.py`
- Select method, sample size, model
- LIME explanation gets created and shown in the notebook as html

## Results (full data set - 0.8/0.2 split):

### Bag of Words:

- Logistic Regression (Bag of Words) Accuracy: 0.8738
- Random Forest (Bag of Words) Accuracy: 0.8495
- SVM (Bag of Words) Accuracy: 0.8719

### TF-IDF:

- Logistic Regression (TF-IDF) Accuracy: 0.8934
- Random Forest (TF-IDF) Accuracy: 0.8513
- SVM (TF-IDF) Accuracy: 0.8991
  
### Word2Vec:
Logistic Regression Accuracy: 0.8443
Logistic Regression Precision: 0.8420425193721438
Logistic Regression Recall: 0.8476

Random Forest Accuracy: 0.805
Random Forest Precision: 0.801621835443038
Random Forest Recall: 0.8106

SVM Accuracy: 0.8571
SVM Precision: 0.8534943575529598
SVM Recall: 0.8622


### Topic Modeling:

- Sentiment Classification Accuracy: 0.8576

                    precision    recall  f1-score   support
          negative       0.86      0.86      0.86      4942
          positive       0.86      0.86      0.86      5058
      
          accuracy                           0.86     10000
         macro avg       0.86      0.86      0.86     10000
      weighted avg       0.86      0.86      0.86     10000


### K-means Clustering (not functional):
                  precision    recall  f1-score   support
    
        negative       0.00      0.00      0.00      4987
        positive       0.50      1.00      0.67      5013
    
        accuracy                           0.50     10000
       macro avg       0.25      0.50      0.33     10000
    weighted avg       0.25      0.50      0.33     10000



#### Observations:

- SVM always seems to take the longest out of the options, while logistic regression is decently fast and provides basically the same accuracy.
- Random forest performs the worst.

## Issues:

- Clustering seems to behave weirdly for now. Unsure if this is normal or if the code is wrong.

## To-do:

- Improve Preprocessing (Add Lemmatization and see if stemming/lemmatization/stemming + lematization works best) - IN PROGRESS!

-  Check what is NER and implement it - IN PROGRESS!


 ## What is important for presentation (Stuff he mentioned):

- Word clouds (for positive and negative decisions)
- Clustering 
- Classification including confusion matrix
- T.SNE
