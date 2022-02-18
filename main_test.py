import streamlit as st
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
##
import math
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# preprocessing
import re
from sklearn.feature_extraction.text import CountVectorizer
import helper_functions as hf
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

##
### custom cmponents
def box_danger(text):
    st.markdown(f'<h2 style="background-color:#850505;color:#fff;font-size:20px;border-radius:10px;padding:20px;text-align:center">{text}</h2>', unsafe_allow_html=True)
    
def box_sucess(text):
    st.markdown(f'<h2 style="background-color:#00a86b;color:#fff;font-size:20px;border-radius:10px;padding:20px;text-align:center">{text}</h2>', unsafe_allow_html=True)
##
##
def tweet_card(text):
     st.markdown(f'<div class="card" style=" box-shadow: 0 4px 8px 0 rgba(0,0,0.2);transition: 0.3s;width: 80%;padding:5%"><div class="container"><p>{text}</p></div></div>', unsafe_allow_html=True) 
##
st.title("Twitter Sentiment Analysis")
st.write("**Detecting Hate Speech**")
st.write("**AI project By Lalit and Akshat**")
st.markdown("<h3>Defination </h3> ",unsafe_allow_html=True)
st.markdown("**You may not promote violence against or directly attack or threaten other people based on race, ethnicity, national origin, caste, sexual orientation, gender, gender identity, religious affiliation, age, disability, or serious disease. **")
##
sentence = st.text_input("Enter a post")
# st.write(df.tweet)
tweets = [sentence]   
cleaned_sentence = []
for i in tweets :
    cleaned_sentence.append(hf.clean_text_round1(i))
# st.write(cleaned_sentence)
# st.write("After cleaning : ",cleaned_sentence)
processed_sentence = []
for i in cleaned_sentence :
    processed_sentence.append(hf.process_tweet(i))
# # processed_sentence = hf.process_tweet(cleaned_sentence)
# # st.write("After NLP : ",processed_sentence)
list_string = []
for i in processed_sentence :
    list_string.append(hf.listToString(i))
# st.write(list_string[2])
# # s = [list_string]
with open('count_vectorizer.pkl' , 'rb') as f:
    vec = pickle.load(f)
s = list_string

# st.write(temp.toarray().shape)

st.sidebar.title("Choose Options")
v_c = st.sidebar.selectbox(
    "Select Vectorization Technique",
    ["Count", "TF-IDF"]
)
m_c = st.sidebar.selectbox(
    "Select Model ",
    ["Naive Bayes", "Logistic Regression"]
)
if v_c == 'Count' :
    with open('count_vectorizer.pkl' , 'rb') as f:
        vec = pickle.load(f)
    temp = vec.transform(s)
    if m_c == 'Naive Bayes' :
        with open('naiveBayes_count_model.pkl','rb') as file :
            model = pickle.load(file)
    else :
        with open('lg_count_model.pkl','rb') as file :
            model = pickle.load(file)
else :
    with open('tfidf_vectorizer.pkl' , 'rb') as f:
        vec = pickle.load(f)
        temp = vec.transform(s)
    if m_c == 'Naive Bayes' :
        with open('naiveBayes_tfidf_model.pkl','rb') as file :
            model = pickle.load(file)
    else :
        with open('lg_tfidf_model.pkl','rb') as file :
            model = pickle.load(file)

re = model.predict(temp)

if st.button("Analyze") :
    if re[0] == 1 :
        box_danger("Potential Hate/Offensive")
    else :
        box_sucess("Looks Fine.")
