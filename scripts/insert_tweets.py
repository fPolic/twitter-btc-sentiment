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


def get_hastag_list(hashtags):
    '''
      split string into list and remove `#`
    '''
    if hashtags == '' or hashtags == None:
        return []
    return [re.sub(r'#([^\s]+)', r'\1', ht) for ht in hashtags.split(' ')]


def get_mentions_list(mentions):
    if mentions == '' or mentions == '@' or mentions == None:
        return []
    return [re.sub(r'@([^\s]+)', r'\1', m) for m in mentions.split(' ')]


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]
    db.tweets.drop()

    file_list = sorted(os.listdir(TWEETS_DIR))
    bar = Bar('Inserting tweets: ', max=len(file_list))

    time_limit = datetime.datetime(2018, 1, 23, 19, 00)
    for path in file_list:
        file_path = os.path.join(TWEETS_DIR, path)

        with open(file_path, mode="r") as csv_file:

            insert_data = []
            file = csv.DictReader(csv_file, delimiter=";")

            for line in file:
                # there are some "empty" tweets in csv files
                if line.get('text') == 'None':
                    continue

                try:
                    parsed_date = datetime.datetime.strptime(
                        line.get('date'), '%Y-%m-%d %H:%M:%S')
                except:
                    parsed_date = datetime.datetime.strptime(
                        line.get('date'), '%Y-%m-%d %H:%M')

                # if parsed_date < time_limit:
                #     continue

                tweet = {
                    "date": parsed_date,
                    "text": line.get('text'),
                    "username": line.get('username'),
                    'retweet': int(line.get('retweets')),
                    'favorites': int(line.get('favorites')),
                    "tokens":  tokenize_tweet(line.get('text')),
                    "hashtags": get_hastag_list(line.get('hashtags')),
                    'mentions': get_mentions_list(line.get('mentions'))
                }

                insert_data.append(tweet)
            bar.next()
            if len(insert_data) > 0:
                db.tweets.insert_many(insert_data)
    bar.finish()


if __name__ == "__main__":
    main()
