import os
import re
import csv
import pprint

from textblob import TextBlob
from string import punctuation
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

DIR = './data/'
DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'

# TODO:
# 1. emojis to list
# 2. `u` (short of YOU)
# 3. stock/coin tickers as $SPY
__stopwords = set(stopwords.words('english') + list(punctuation))


def process_tweet(tweet):
    # print(tweet)
    # lowercase
    tweet = tweet.lower()
    # remove URLS
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',
                   '', tweet)
    # remove usernames
    tweet = re.sub('@[^\s]+', '', tweet)
    # remove the # in #hashtag
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    # remove repeated characters (helloooooooo into hello)
    tweet = word_tokenize(tweet)
    # list of words without stop words
    return [word for word in tweet if word not in __stopwords]


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]

    tweets = db.tweets.find().limit(10)
    for tweet in tweets:

        tokens = process_tweet(tweet.get('text'))
        anlysis = TextBlob(' '.join(tokens))
        sentiment_polarity = anlysis.sentiment.polarity

        db.tweets.update_one(tweet, {"$set": {
            "tokenized": tokens,
            "sentiment": sentiment_polarity
        }})


if __name__ == "__main__":
    main()
