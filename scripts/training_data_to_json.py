import re
import csv
import json

from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

__stopwords = set(stopwords.words('english') + list(punctuation))


def process_tweet(tweet):
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


if __name__ == '__main__':
    dataset_to_json()
