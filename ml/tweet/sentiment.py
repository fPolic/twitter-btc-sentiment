import os
import re
import csv
import pprint
import pickle
import pandas as pd

from progress.bar import Bar
from textblob import TextBlob
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples

from ml.helpers.mongo import get_tweet_repo
from ml.helpers.lexicon import get_emo_nrc_lexicon
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vaderAnalyzer = SentimentIntensityAnalyzer()

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
    """
     Get tweet sentiment using pretrained NB classifier
    """
    return __NaiveBayesClassifier.classify(text)


def analyze_sentiment(text):
    """
      Get tweet sentiment using builtin analyser from TextBlob
    """
    analysis = TextBlob(text)
    # skip objective sentences(facts, infos, etc.)
    if analysis.sentiment.subjectivity < 0.5:
        return None

    return analysis.sentiment.polarity


def process_tweets_sentiment(count=200):
    """
      Claculate sentiment for first `count` tweets from colection.
    """
    tweets = TweetRepo.find().limit(count)

    for tweet in tweets:
        # do we want to use `hashtags` here
        raw_text = tweet.get('text')
        # + ' ' + ' '.join(tweet.get('hashtags')) // SHOULD WE USE HASHTAGS?
        tokens = tokenize_tweet(raw_text)
        text = ' '.join(tokens)

        TweetRepo.update_one(tweet, {"$set": {
            "sentiment": analyze_sentiment(text),
            "classification": classify_sentiment(text)
        }})


def process_tweets_emotions(count=200):
    """
      Claculate sentiment for first `count` tweets from colection.
    """
    tweets = TweetRepo.find()  # .limit(count)
    lexicon = get_emo_nrc_lexicon()
    bar = Bar('Processing emotions', max=tweets.count())

    EMOTIONS = [
        'anger',
        'anticipation',
        'disgust',
        'fear',
        'joy',
        'negative',
        'positive',
        'sadness',
        'surprise',
        'trust'
    ]

    for tweet in tweets:
        insert_em = {}
        raw_text = tweet.get('text')
        tokens = tokenize_tweet(raw_text)
        tokens_df = pd.DataFrame({'word': tokens})
        # left join emotions on word/token
        token_emotions = pd.merge(tokens_df, lexicon,  on='word', how='left')
        # make only `emotions` projection
        emotions = token_emotions[EMOTIONS].sum(axis=0, skipna=True).items()
        # build record data
        for em, acc in emotions:
            insert_em[em] = int(acc)

        TweetRepo.update_one(tweet, {"$set": insert_em})
        bar.next()
        # use this sentiment socre to determine which tweets should be considered ???
        # print(vaderAnalyzer.polarity_scores(
        #     raw_text + ' ' + ' '.join(tweet.get('hashtags'))))
    bar.finish()
