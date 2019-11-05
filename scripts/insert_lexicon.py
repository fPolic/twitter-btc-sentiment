import os
import csv
import json

from ml.helpers.lexicon import get_emo_nrc_lexicon
from datetime import datetime
from pymongo import MongoClient

DIR = './data/'
DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]
    db.lexicon.drop()

    lexicon = get_emo_nrc_lexicon()
    records = json.loads(lexicon.T.to_json()).values()
    db.lexicon.insert_many(records)


if __name__ == "__main__":
    main()
