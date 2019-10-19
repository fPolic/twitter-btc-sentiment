import os
import re
import csv
import pprint

from textblob import TextBlob
from string import punctuation
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'
# use single global connection
client = MongoClient(mongoURL)
db = client[DATABASE_NAME]

# =============== PROCESSING ===============

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

# =============== STRATEGIES ===============


def process_corpus_sentiment(tweets):
    for tweet in tweets:
        tokens = process_tweet(tweet.get('text') + ' ' +
                               ' '.join(tweet.get('hashtags')))  # do we want to use `hastags` here
        analysis = TextBlob(' '.join(tokens))

        # skip objective sentences (facts, infos, etc.)
        if analysis.sentiment.subjectivity < 0.5:
            continue

        sentiment_polarity = analysis.sentiment.polarity

        db.tweets.update_one(tweet, {"$set": {
            "tokenized": tokens,  # TODO: don't save this
            "sentiment": sentiment_polarity
        }})


def main():
    tweets = db.tweets.find().limit(300)
    process_corpus_sentiment(tweets)


if __name__ == "__main__":
    main()
