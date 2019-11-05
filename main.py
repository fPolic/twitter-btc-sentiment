from ml.tweet.sentiment import process_tweets_emotions
from ml.tweet.aggregate import aggregate_tweet_emotions_by_date, create_emotions_aggregation_view

# from scripts.insert_tweets import main

if __name__ == '__main__':
    # process_tweets_emotions()
    # aggregate_tweet_emotions_by_date()
    create_emotions_aggregation_view()
    # main()
