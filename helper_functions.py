# importing relevant libraries
import numpy as np
import pandas as pd
import pickle

# NLP libraries
import re
import string
import nltk
from sklearn.feature_extraction import text 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

## functions used in data_cleaning.ipynb

# tweet cleaning function
def clean_text_round1(text):
    '''Make text lowercase, remove punctuation, mentions, hashtags and words containing numbers.'''
    # make text lowercase
    text = text.lower()
    # removing text within brackets
    text = re.sub('\[.*?\]', '', text)
    # removing text within parentheses
    text = re.sub('\(.*?\)', '', text)
    # removing numbers
    text = re.sub('\w*\d\w*', '', text)
    # if there's more than 1 whitespace, then make it just 1
    text = re.sub('\s+', ' ', text)
    # if there's a new line, then make it a whitespace
    text = re.sub('\n', ' ', text)
    # removing any quotes
    text = re.sub('\"+', '', text)
    # removing &amp;
    text = re.sub('(\&amp\;)', '', text)
    # removing any usernames
    text = re.sub('(@[^\s]+)', '', text)
    # removing any hashtags
    text = re.sub('(#[^\s]+)', '', text)
    # remove `rt` for retweet
    text = re.sub('(rt)', '', text)
    # string.punctuation is a string of all punctuation marks
    # so this gets rid of all punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # getting rid of `httptco`
    text = re.sub('(httptco)', '', text)
    return text


## functions used in nlp_preprocessing.ipynb

def unfiltered_tokens(text):
    """tokenizing without removing stop words"""
    dirty_tokens = nltk.word_tokenize(text)
    return dirty_tokens

# tokenizing and removing stop words
def process_tweet(text):
    """tokenize text in each column and remove stop words"""
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
    return stopwords_removed 

#### list to string
def listToString(s): 
    
    # initialize an empty string 
    string = " "
    
    # return string 
    return (string.join(s)) 


##### function for lemetization
def lemmatization(processdata) :
    lemmatizer = WordNetLemmatizer() 
    lemmatized_output = []

    for listy in processdata:
        lemmed = ' '.join([lemmatizer.lemmatize(w) for w in listy])
        lemmatized_output.append(lemmed)
    return lemmatized_output
## functions used for modeling process
