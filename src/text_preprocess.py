import string
import re
import spacy
from nltk.corpus import stopwords

def filter_punctuation(texts):
    """
    Makes texts lowercase, deletes punctuation, and double spaces
    
     # Arguments:
        texts: a list of the texts
        
    # Returns:
        list of the filtered texts 
    """
    return [re.sub('\s+', ' ', text.translate("".maketrans(string.punctuation, " "*len(string.punctuation)))).strip().lower() for text in texts]

def en_lemmatize(texts):
    """
    Lemmatize english texts.
    
     # Arguments:
        texts: a list of the texts
        
    # Returns:
        list of the lemmatized texts 
    """
    nlp_model = spacy.load("en_core_web_md", disable=['parser', 'ner'])
    return [" ".join([token.lemma_ for token in nlp_model(text)]) for text in texts]

def filter_stopwords(texts, additional_stopword=[]):
    """
    Filter typical  english stop-words from nltk.corpus stop-words list.
    
    # Arguments:
        texts: a list of the texts
        
    # Returns:
        list of the filtered texts 
    """
    stop_words = stopwords.words("english")+additional_stopwords
    return [" ".join([word for word in text.split() if word not in stop_words]) for text in texts]