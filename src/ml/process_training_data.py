import csv
import json

from process_tweets import process_tweet
from textblob.classifiers import NaiveBayesClassifier


def dataset_to_json():
    with open('../../data/Sentiment_Analysis_Dataset.csv', 'r') as csvdata:
        reader = csv.DictReader(csvdata)
        data = []
        i = 0
        for row in reader:
            i += 1
            label = 'neg' if row.get('Sentiment') == '0' else 'pos'
            text = ' '.join(process_tweet(row.get('SentimentText')))

            data.append({
                "text": text,
                "label": label
            })
            if i > 1000:
                break
    with open("../../data/nb-data.json", 'w') as f:
        json.dump(data, f)


dataset_to_json()
