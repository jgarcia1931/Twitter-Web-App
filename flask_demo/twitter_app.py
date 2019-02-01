#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)
#######################################################################################
#
# Functions
#
#######################################################################################
# General:
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
import math
# We import our access keys:
from credentials import *  # This will allow us to use the keys as variables

import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')
from nltk.tokenize import TweetTokenizer

from textblob import TextBlob
import re

# Happy Emoticons
emoticons_happy = set([':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
                       ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
                       '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
                       'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)','<3'])
# Sad Emoticons
emoticons_sad = set([':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
                     ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
                     ':c', ':{', '>:\\', ';('])
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

def word_of_interest(data, word):
    rows = len(data)
    rows_of_interest = []
    for row in range(rows):
        word_list = data['Tweets_clean'][row]
        #     print(word_list)
        if len(word_list) < 1:
            continue
        else:
            for i in pd.Series(word_list).str.contains(word):
                if i == True:
                    rows_of_interest.append(row)
                    continue
    data = data.iloc[rows_of_interest, :]
    data.reset_index(drop=True, inplace=True)

    return data

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tweet)
    result = analysis.sentiment.polarity
    return result

def clean_tweets(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'[\s]RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)

    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in emoticons and  # remove emoticons
                word not in string.punctuation):  # remove punctuation
            tweets_clean.append(word)
    #             stem_word = stemmer.stem(word) # stemming word
    #             tweets_clean.append(stem_word)

    return tweets_clean

def pull_tweets(handle):

    # API's setup:
    def twitter_setup():
        """
        Utility function to setup the Twitter's API
        with our access keys provided.
        """
        # Authentication and access using keys:
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

        # Return API with authentication:
        api = tweepy.API(auth)
        return api

    # We create an extractor object:
    extractor = twitter_setup()
    handles = [handle]
    for Handle in handles:
        try:
            total_tweets = extractor.get_user(Handle).statuses_count
        except:
            data = "@" + Handle + " does not exist"
            return data

        if total_tweets == 0:
            data = "@" + Handle + " account has no tweets"
            return data

        start_date = extractor.get_user(Handle).created_at
        print(total_tweets)
        print(start_date)
        iters = math.ceil(total_tweets/200)
        full_iters = math.floor(total_tweets/200)
        remainder = total_tweets%200
        data = pd.DataFrame()

        # tweets_temp = tweets
        tweets_temp = []

        for its in range(iters):

            if total_tweets <= 0:
                continue

            total_count = total_tweets

            if its > 0:
                total_count = total_count + 1


            if len(tweets_temp) == 0:
                try:
                    tweets_temp = extractor.user_timeline(screen_name=Handle, count=total_count)
                except:
                    data = "@" + Handle + " might be a private account"
                    return data
            else:
                old_id = tweets_temp[ len(tweets_temp)-1 ].id
                tweets_temp = extractor.user_timeline(screen_name=Handle, count=total_count, max_id=old_id)

            data2 = pd.DataFrame(data=[tweet.text for tweet in tweets_temp], columns=['Tweets'])
            data2['len'] = np.array([len(tweet.text) for tweet in tweets_temp])
            data2['ID'] = np.array([tweet.id for tweet in tweets_temp])
            data2['Date'] = np.array([tweet.created_at for tweet in tweets_temp])
    #         data2['Source'] = np.array([tweet.source for tweet in tweets_temp])
            data2['Likes'] = np.array([tweet.favorite_count for tweet in tweets_temp])
            data2['RTs'] = np.array([tweet.retweet_count for tweet in tweets_temp])
    #         data2['Geo'] = np.array([tweet.geo for tweet in tweets_temp])

            data = pd.concat([data, data2])
            data.reset_index(drop=True, inplace=True)

            if len(data) > 3200:
                break

            data.drop(data.index[len(data) - 1])


            total_tweets = total_tweets - 200
    data.drop_duplicates(subset='Tweets', inplace=True)
    return data


@app.route('/', methods = ['GET','POST'])
def index2():

    if request.method == 'GET':
        return render_template('index2.html')

    if request.method == 'POST':

        if 'twitter_handle' in request.form:
            handle = request.form['twitter_handle']
            num_results = request.form['num_results']
            twitter_method = request.form.getlist("twitter_method")

            data = pull_tweets(handle)
            if type(data) == str:
                table_true = 2
                return render_template('index2.html', table_op=str(table_true), handle=handle, data_str=data)
            else:
                data.reset_index(drop=True, inplace=True)
                rows = len(data)
                tokened = []
                for row in range(rows):
                    word_list = clean_tweets(data.loc[row, 'Tweets'])
                    tokened.append(word_list)
                data['Tweets_clean'] = pd.Series(tokened)

                if twitter_method[0] == 'option1':
                    data_final = data[['Tweets', 'ID', 'Date', 'Tweets_clean']][:int(num_results)]

                if twitter_method[0] == 'option2':
                    data['Tweet_str'] = data['Tweets_clean'].apply(lambda x: ' '.join(x))
                    data['SA'] = np.array([analize_sentiment(tweet) for tweet in data['Tweet_str']])

                    tweets_pos = []
                    data.reset_index(drop=True,inplace=True)
                    for row in range(len(data)):
                        if data.loc[row, 'SA'] < -.5:
                            tweets_pos.append(row)
                    data_final = data.iloc[tweets_pos, :][['Tweets', 'ID', 'Date', 'Tweets_clean']]
                    data_final.reset_index(drop=True,inplace=True)
                word_search = "none"

        if 'twitter_hand' in request.form:
            handle = request.form['twitter_hand']
            word_search = request.form['word_search']

            data = pull_tweets(handle)

            if type(data) == str:
                table_true = 2
                return render_template('index2.html', table_op=str(table_true), handle=handle, data_str=data)
            else:
                data.reset_index(drop=True, inplace=True)
                rows = len(data)
                tokened = []
                for row in range(rows):
                    word_list = clean_tweets(data.loc[row, 'Tweets'])
                    tokened.append(word_list)
                data['Tweets_clean'] = pd.Series(tokened)

                data_final = word_of_interest(data, word_search)

        if len(data_final) > 0:
            table_true = 1
        else:
            table_true = 0


        return render_template('index2.html', table_op=str(table_true),
                               data=data_final.to_html(classes="table table-striped"), handle=handle,
                               word_search=word_search)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)








