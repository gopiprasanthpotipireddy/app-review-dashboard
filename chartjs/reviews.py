from calendar import month
from itertools import count
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer  # For Bag of words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from operator import ne, neg
from turtle import pos
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
import datetime as DT
from dateutil.relativedelta import relativedelta
from collections import Counter

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class Review:

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def remove_emoji(self, string):
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

    def token(self, text):
        word_tokens = word_tokenize(text)
        return word_tokens

    def remove_stopwords(self, words):
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    def stem_words(self, words):
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize(self, words):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def jaccard(self, str1, str2):
        a = set(str(str1).split())
        b = set(str(str2).split())
        c = a.intersection(b)
        return list(c)

    def sentence(self, text):
        sentence = " ".join(text)
        return sentence

    def year(self, time):
        return str(time).split("-")[0]

    def year_and_month(self, time):
        date = str(time).split(" ")[0]
        dateArray = str(date).split("-")
        return dateArray[0]+"-"+dateArray[1]
    def month(self, time):
        dateArray = str(time).split("-")
        return dateArray[1]

    def date(self, time):
        return str(time).split(" ")[0]

    def get_all_reviews(self, request):

        app_name = request.GET.get('q')
        result = reviews_all(
            app_name,
            sleep_milliseconds=0,  # defaults to 0
            lang='en',            # defaults to 'en'
            country='in',         # defaults to 'us'
            # defaults to Sort.MOST_RELEVANT , you can use Sort.NEWEST to get newst reviews
            sort=Sort.NEWEST,
        )
        data = pd.DataFrame(np.array(result), columns=['review'])
        data = data.join(pd.DataFrame(data.pop('review').tolist()))
        response=self.get_sentiment_from_data(data)
        return response

    def get_sentiment_from_data(self, data, months,from_date):
        data["content"] = data["content"].apply(
            lambda x: self.clean_text(str(x)))
        data["content"] = data["content"].apply(
            lambda x: self.remove_emoji(str(x)))
        data["reviews"] = data["content"].apply(lambda x:  self.token(x))
        data["reviews"] = data["reviews"].apply(
            lambda x: self.remove_stopwords(x))
        data["selected_text"] = data["reviews"].apply(
            lambda x: self.stem_words(x))
        data["selected_text"] = data["selected_text"].apply(
            lambda x: self.lemmatize(x))
        data = data.drop(columns=['reviewId', 'userName', 'userImage',
                                  'thumbsUpCount', 'reviewCreatedVersion', 'replyContent',
                                  'repliedAt'], axis=1)
        data["selected_text"] = data["selected_text"].apply(
            lambda x: self.sentence(x))
        data["reviews"] = data["reviews"].apply(lambda x: self.sentence(x))
        sentiment = []
        sid_obj = SentimentIntensityAnalyzer()
        for i in data["selected_text"]:
            sentiment_dict = sid_obj.polarity_scores(i)
            if sentiment_dict['compound'] >= 0.05:
                sentiment.append("Positive")

            elif sentiment_dict['compound'] <= - 0.05:
                sentiment.append("Negative")

            else:
                sentiment.append("Neutral")

        data["sentiment"] = sentiment

        # Overall Percentage by Sentiment
        counts = data.groupby(['sentiment'])['sentiment']
        total = len(data.index)

        # Filter last 7 days reviews
        data_last_7days = self.get_last_7_days(data)
         # Filter last 7 months reviews
        data_last_7months=None
        if(months>0):
            data_last_7months = self.get_last_n_months(data, from_date)

        # Top 20 Common words
        data['temp_list'] = data["reviews"].apply(lambda x: str(x).split())
        data_pos = data[data["sentiment"] == "Positive"]
        data_neg = data[data["sentiment"] == "Negative"]
        data_Neu = data[data["sentiment"] == "Neutral"]

        data_pos['temp_list'] = data_pos["reviews"].apply(
            lambda x: str(x).split())
        top_pos = Counter([item for item in data_pos['temp_list']
                          for item in item])
        temp_pos = pd.DataFrame(top_pos.most_common(20))
        temp_pos.columns = ['common_word', 'count']
        data_neg['temp_list'] = data_neg["reviews"].apply(
            lambda x: str(x).split())
        top_neg = Counter([item for item in data_neg['temp_list']
                          for item in item])
        temp_neg = pd.DataFrame(top_neg.most_common(20))
        temp_neg.columns = ['common_word', 'count']
        data_Neu['temp_list'] = data_Neu["reviews"].apply(
            lambda x: str(x).split())
        top_Neu = Counter([item for item in data_Neu['temp_list']
                          for item in item])
        temp_Neu = pd.DataFrame(top_Neu.most_common(20))
        temp_Neu.columns = ['common_word', 'count']

        top = Counter([item for item in data['temp_list'] for item in item])
        common_words = pd.DataFrame(top.most_common(20))
        common_words.columns = ['common_word', 'count']
        result_all = json.loads(common_words.to_json(orient="table"))
        result_pos = json.loads(temp_pos.to_json(orient="table"))
        result_neg = json.loads(temp_neg.to_json(orient="table"))
        result_neu = json.loads(temp_Neu.to_json(orient="table"))

        response = {
            'overall': {
                'labels': ['Positive', 'Negative', 'Neutral'],
                'percentage': [(counts.get_group("Positive").count()/total)*100,
                               (counts.get_group("Negative").count()/total)*100,
                               (counts.get_group('Neutral').count()/total)*100
                               ]
            },
            'last7Days': data_last_7days,
            'last7Months': data_last_7months,
            'common_words': {
                "all": result_all["data"],
                "pos": result_pos["data"],
                "neg": result_neg["data"],
                "neu": result_neu["data"]
            }
        }
        return response

    def get_start_date_time(self, input_date):
        return DT.datetime.combine(input_date, DT.datetime.min.time())

    def get_last_n_months(self,data,from_date):
        data['month'] = data['at'].apply(lambda x: self.year_and_month(x))
        today = DT.datetime.utcnow().date()
        months_list=[]

        while(True):
            if(self.year(from_date)>self.year(today)):
                break
            if(self.year(from_date)==self.year(today) and self.month(from_date)>self.month(today)):
                break
            months_list.append(self.year_and_month(from_date))
            from_date = from_date+relativedelta(months=1)
        
        last_n_months = data.groupby(['sentiment', 'month'])
        positive = []
        negative = []
        neutral = []
        keys = last_n_months.groups.keys()

        for sub_ind in months_list:
            if ('Positive', sub_ind) in keys:
                positive.append(last_n_months.get_group(
                    ('Positive', sub_ind)).size)
            else:
                positive.append(0)
            if ('Negative', sub_ind) in keys:
                negative.append(last_n_months.get_group(
                    ('Negative', sub_ind)).size)
            else:
                negative.append(0)
            if ('Neutral', sub_ind) in keys:
                neutral.append(last_n_months.get_group(('Neutral', sub_ind)).size)
            else:
                neutral.append(0)
        return  {
                'labels': months_list,
                'chartdata': {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral,
                }
            }


    def get_last_7_days(self,data):
        today = DT.datetime.utcnow().date()
        dt = DT.datetime.combine(today, DT.datetime.min.time())
        dates_list = []
        past_time = dt - DT.timedelta(days=7)
        for i in range(6, -1, -1):
            dates_list.append(str(today - DT.timedelta(days=i)))

        reviews_last7 = data.loc[data['at'] > past_time]
        reviews_last7['date'] = reviews_last7['at'].apply(
            lambda x: self.date(x))
        last7 = reviews_last7.groupby(['sentiment', 'date'])
        positive = []
        negative = []
        neutral = []
        keys = last7.groups.keys()

        for sub_ind in dates_list:
            if ('Positive', sub_ind) in keys:
                positive.append(last7.get_group(
                    ('Positive', sub_ind)).size)
            else:
                positive.append(0)
            if ('Negative', sub_ind) in keys:
                negative.append(last7.get_group(
                    ('Negative', sub_ind)).size)
            else:
                negative.append(0)
            if ('Neutral', sub_ind) in keys:
                neutral.append(last7.get_group(('Neutral', sub_ind)).size)
            else:
                neutral.append(0)
        return  {
                'labels': dates_list,
                'chartdata': {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral,
                }
            }

    def get_reviews_from_date(self, request):

        app_name = request.GET.get('q')
        days=request.GET.get('days')
        months=request.GET.get('months')

        today = DT.datetime.utcnow().date()
        dt = DT.datetime.combine(today, DT.datetime.min.time())

        if(days):
            from_date = dt - DT.timedelta(days=int(days))
            months=0
        if(months):
            from_date=DT.datetime(int(self.year(dt)),int(self.month(dt)),1)
            from_date=from_date-relativedelta(months=int(months))

        continuation_token = None
        complete_data = list()
        while(True):
            result, continuation_token = reviews(
                app_name,
                lang='en',  # defaults to 'en'
                country='us',  # defaults to 'us'
                sort=Sort.NEWEST,  # defaults to Sort.NEWEST,
                continuation_token=continuation_token,
                count=100
            )
            if(len(result)==0):
                break

            first_row = result[0]
            last_row = result[-1]

            if(first_row['at'] < from_date):
                break
            if(last_row['at'] < from_date):
                result = list(filter(lambda x: x['at'] >= from_date, result))
                complete_data = complete_data+result
                break
            complete_data = complete_data+result

        data = pd.DataFrame(np.array(complete_data), columns=['review'])
        data = data.join(pd.DataFrame(data.pop('review').tolist()))
        response=self.get_sentiment_from_data(data, int(months),from_date)
        return response
