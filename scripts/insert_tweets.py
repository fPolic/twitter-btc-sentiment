import os
import csv
import datetime

from pymongo import MongoClient

DATABASE_NAME = 'fer'
TWEETS_DIR = './data/tweets/'
mongoURL = 'mongodb://localhost:27017/'


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]
    db.tweets.drop()

    file_list = os.listdir(TWEETS_DIR)
    for path in file_list[:3]:

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
                    "username": line.get('username'),
                    "text": line.get('text'),
                    "hashtags": line.get('hashtags'),
                }

                insert_data.append(tweet)
            db.tweets.insert_many(insert_data)


if __name__ == "__main__":
    main()
