import os
import re
import csv
import pprint
import pickle

from textblob import TextBlob
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples

from ml.helpers.mongo import get_tweet_repo

__NaiveBayesClassifier = pickle.load(open("naivebayes.pickle", "rb"))
TweetRepo = get_tweet_repo()

# =============== PROCESSING ===============

# TODO:
# 1. emojis to list
# 2. `u` (short of YOU)
# 3. stock/coin tickers as $SPY
__stopwords = set(stopwords.words('english') + list(punctuation))


def tokenize_tweet(tweet):
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
    tokens = word_tokenize(tweet)
    # list of words without stop words
    return [word for word in tokens if word not in __stopwords]

# =============== STRATEGIES ===============


def classify_sentiment(text):
    return __NaiveBayesClassifier.classify(text)


def analyze_sentiment(text):
    analysis = TextBlob(text)

    # skip objective sentences(facts, infos, etc.)
    if analysis.sentiment.subjectivity < 0.5:
        return None

    return analysis.sentiment.polarity


def process_tweets_sentiment(count=100):
    tweets = TweetRepo.find().limit(count)

    for tweet in tweets:
        # do we want to use `hashtags` here
        raw_text = tweet.get('text')  # + ' ' + ' '.join(tweet.get('hashtags'))
        tokens = tokenize_tweet(raw_text)
        text = ' '.join(tokens)

        TweetRepo.update_one(tweet, {"$set": {
            # "tokenized": tokens,  # TODO: don't save this
            "sentiment": analyze_sentiment(text)
        }})

        TweetRepo.update_one(tweet, {"$set": {
            "classification": classify_sentiment(text)
        }})
