from bs4 import BeautifulSoup
import re
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
import nltk

def preprocess_text(text):
    tokenizer = ToktokTokenizer()
    stopword_list = stopwords.words('english')
    
    # Remove unwanted stuff like special characters, brackets, etc.
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\bbr\b', '', text)
    
    # Tokenize and remove stopwords
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    
    # Stemming
    ps = nltk.porter.PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    
    return ' '.join(stemmed_tokens)