import os
import re
import csv
import datetime

from progress.bar import Bar
from pymongo import MongoClient
from ml.tweet.sentiment import tokenize_tweet

DATABASE_NAME = 'fer'
TWEETS_DIR = './data/tweets/'
mongoURL = 'mongodb://localhost:27017/'

# split string into list and remove `#`


def get_hastag_list(hashtags):
    if hashtags == None:
        return []
    return [re.sub(r'#([^\s]+)', r'\1', ht) for ht in hashtags.split(' ')]


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]
    db.tweets.drop()

    file_list = sorted(os.listdir(TWEETS_DIR))[30:36]
    bar = Bar('Inserting tweets: ', max=len(file_list))
    for path in file_list:
        file_path = os.path.join(TWEETS_DIR, path)

        with open(file_path, mode="r") as csv_file:

            insert_data = []
            file = csv.DictReader(csv_file, delimiter=";")

            for line in file:
                try:
                    parsed_date = datetime.datetime.strptime(
                        line.get('date'), '%Y-%m-%d %H:%M:%S')
                except:
                    parsed_date = datetime.datetime.strptime(
                        line.get('date'), '%Y-%m-%d %H:%M')
                tweet = {
                    "date": parsed_date,
                    # "username": line.get('username'),
                    "text": line.get('text'),
                    "tokens":  tokenize_tweet(line.get('text')),
                    "hashtags": get_hastag_list(line.get('hashtags')),
                    # do we care about this?
                    # we can eliminate duplicate tweets with Mongo index
                    # 'retweet': int(line.get('retweets'))
                }

                # print(tweet)
                # return
                insert_data.append(tweet)
            bar.next()
            db.tweets.insert_many(insert_data)
    bar.finish()


if __name__ == "__main__":
    main()
