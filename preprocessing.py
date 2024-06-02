from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer

def preprocess_text(text):
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')

    # Match and remove noisy text
    text = BeautifulSoup(text, "html.parser").get_text() # remove html signal like <br><\br>
    text = re.sub(r'\b\w*@\w+\b','', text) # remove unstructure words like @username, F@ck
    text = re.sub(r'http\S+', '', text) # remove website http...
    text = re.sub(r'\[[^]]\]', '', text) # remove square brackets: [....]
    text = re.sub(r'[^a-zA-z0-9\s]', '', text) # remove special characters: @#&*()

    # Tokenize and remove stopwords
    tokens = tokenizer.tokenize(text)
    filter_tokens = [token for token in tokens if token.lower() not in stopword_list]
    
    # Stemming
    ps = PorterStemmer()
    # ps = SnowballStemmer()
    stemmed_tokens = [ps.stem(filter_token) for filter_token in filter_tokens]

    # Lemmatization
    lemmatizer=WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(stemmed_token) for stemmed_token in stemmed_tokens]

    return ' '.join(lemmatized_tokens)