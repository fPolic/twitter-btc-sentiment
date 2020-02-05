from ml.tweet.sentiment import process_tweets_emotions
from ml.tweet.aggregate import aggregate_tweet_emotions_by_houre
from ml.tweet.join_time_series import join_time_series
from ml.tweet.word_count_generate import generate_word_count
from ml.tweet.count_distinct_user_tweets import generate_user_post_counts

from scripts.insert_tweets import main as insert_tweets
from scripts.insert_stock_data import main as insert_stock_data

if __name__ == '__main__':
    # generate_user_post_counts()
    # process_tweets_emotions()
    # insert_tweets()
    # aggregate_tweet_emotions_by_houre()
    # join_time_series()
    # generate_word_count()
    # insert_stock_data()
