from ml.tweet.sentiment import process_tweets_emotions
from ml.tweet.aggregate import aggregate_tweet_emotions_by_houre

from scripts.insert_tweets import main as insert_tweets

if __name__ == '__main__':
    # process_tweets_emotions()
    # insert_tweets()
    aggregate_tweet_emotions_by_houre()
