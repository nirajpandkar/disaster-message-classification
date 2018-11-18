import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re

def tokenize(text):
    """
    Generates tokens from a given document
    
    Arguments:
        text (str): Document to be tokenized
    
    Returns:
        tokens: A list of tokens
    """
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if bool(re.search(r"[^a-zA-Z0-9]", w)) != True]
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens if stopwords.words("english")]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    return tokens

