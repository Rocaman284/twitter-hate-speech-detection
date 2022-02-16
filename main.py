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
import twint
c = twint.Config()
import pandas as pd
import os
from wordcloud import WordCloud
# import only system from os
from os import system, name
from pathlib import Path

##
### custom cmponents
def box_danger(text):
    st.markdown(f'<h2 style="background-color:#850505;color:#fff;font-size:20px;border-radius:10px;padding:20px;text-align:center">{text}</h2>', unsafe_allow_html=True)
    
def box_sucess(text):
    st.markdown(f'<h2 style="background-color:#00a86b;color:#fff;font-size:20px;border-radius:10px;padding:20px;text-align:center">{text}</h2>', unsafe_allow_html=True)
##
def clear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
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
completed = False
sentence = st.text_input("Enter a hashtag, username, to scrap tweet ")

if st.button("Analyse") :
    print("-------------------------------------")
    c.Search =sentence
    c.StoreObject = True
    c.Limit = 100
    c.Store_csv = True
    c.Output = sentence+".csv"
    completed = True
    c.Lang = "en"
    twint.run.Search(c)
    # loader = False
# from twitterscraper import query_tweets
# load saved 
if completed :
    filename = sentence+".csv"

    df = pd.read_csv(sentence+".csv")
    clear()
    # st.write(df.tweet)
    tweets = df.tweet
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

    # if os.path.exists(filename):
    #     os.remove(filename)
    re = model.predict(temp)
    Sum = sum(re)
    per = math.floor((Sum/len(re))*100)
    # st.write("Percentage of hate : ", (Sum/len(re))*100)
    
    labels = 'Potential Hate', 'Looks Fine'
    sizes = [per,100-per]
    explode = (0.2, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    color = ['#850505','#00a86b']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=color,autopct='%1.1f%%',shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
    mask = np.array(Image.open('./twitter.png'))
    wc = WordCloud(background_color="white",mask=mask, max_words=2000, max_font_size=256,random_state=42,width=mask.shape[1],height=mask.shape[0])
    wc.generate(''.join(s))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('on')
    st.pyplot(plt)
    # if re[0]== 0 :
    #     box_sucess("Looks Fine")
    # else :
    #     box_danger("Model predicts potential Hate Speech in the Post")
    sid_obj = SentimentIntensityAnalyzer()
    o = sid_obj.polarity_scores(tweets)
    polarity = o['compound']
    if polarity > 0 :
        box_sucess("Tweets have Positivity with score of "+str(polarity))
    else :
        box_danger("Tweets have Negativity with score of "+str(polarity))
    if sentence[0:0] == "#" :
        st.markdown("Take me to tweets [link](https://twitter.com/hashtag/"+sentence[1:len(sentence)])
    else :
        st.markdown("Take me to tweets [link](https://mobile.twitter.com/search?q="+sentence)
    # translator = Translator()
    # for i in tweets :
        
    #     source_lan = "hi"
    #     translated_to= "en" #hi is the code for Hindi Language

    #     #translate text
    #     translated_text = translator.translate(i, src=source_lan, dest = translated_to)
    #     tweet_card(translated_text.text)
