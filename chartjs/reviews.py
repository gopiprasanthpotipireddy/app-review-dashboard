import requests
from bs4 import BeautifulSoup
from google_play_scraper import Sort, reviews_all, reviews, app
import pandas as pd
import numpy as np
import sys
import re
import string
import nltk
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer  


class Review:


    def clean_text(self,text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    
    def remove_emoji(self,string):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    
    def token(self,text):
     word_tokens = word_tokenize(text)
     return  word_tokens



    def remove_stopwords(self,words):
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    def stem_words(self,words):
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems
    def lemmatize(self,words):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas   

    def jaccard(self,str1, str2): 
        a = set(str(str1).split()) 
        b = set(str(str2).split())
        c = a.intersection(b)
        return list(c)

    def sentence(self,text):
     sentence = " ".join(text)
     return sentence
       
    
    def get_all_reviews(self):
        result = reviews_all(
        'com.rails.red',
        sleep_milliseconds=0, # defaults to 0
        lang='en',            # defaults to 'en'
        country='in',         # defaults to 'us'
        sort=Sort.NEWEST,     # defaults to Sort.MOST_RELEVANT , you can use Sort.NEWEST to get newst reviews
        )
        data = pd.DataFrame(np.array(result),columns=['review'])
        data = data.join(pd.DataFrame(data.pop('review').tolist()))
        
        data["content"] = data["content"].apply(lambda x:self.clean_text(str(x)))
        
        data["content"] = data["content"].apply(lambda x:self.remove_emoji(str(x)))
        data["reviews"] = data["content"].apply(lambda x:  self.token(x))
        data["selected_text"] = data["reviews"].apply(lambda x:  self.remove_stopwords(x))
        data["selected_text"]=data["selected_text"].apply(lambda x: self.stem_words(x))
        data["selected_text"]=data["selected_text"].apply(lambda x: self.lemmatize(x))
        data=data.drop(columns=['reviewId', 'userName', 'userImage',
       'thumbsUpCount', 'reviewCreatedVersion', 'replyContent',
       'repliedAt'],axis=1)
        data["selected_text"]=data["selected_text"].apply(lambda x: self.sentence(x))
        data["reviews"]=data["reviews"].apply(lambda x: self.sentence(x))
        sentiment=[]
        sid_obj = SentimentIntensityAnalyzer()
        for i in data["selected_text"]:
            sentiment_dict = sid_obj.polarity_scores(i)
            if sentiment_dict['compound'] >= 0.05 :
                  sentiment.append("Positive")
          
            elif sentiment_dict['compound'] <= - 0.05 :
                   sentiment.append("Negative")
          
            else :
                   sentiment.append("Neutral")

        data["sentiment"]=sentiment
        counts=data.groupby(['sentiment'])['sentiment'];
        total=len(data.index);
        response={'labels':['Positive','Negative','Neutral'],
                  'percentage':[(counts.get_group("Positive").count()/total)*100,
                                (counts.get_group("Negative").count()/total)*100,
                                (counts.get_group('Neutral').count()/total)*100
                               ]  
                }
        return response;












    
