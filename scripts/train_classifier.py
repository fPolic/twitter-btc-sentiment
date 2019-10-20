
import random
import pickle

from textblob.classifiers import NaiveBayesClassifier
from nltk.corpus import twitter_samples


pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

# positive tweets words list
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((tweet, 'pos'))

# negative tweets words list
neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((tweet, 'neg'))

# random.shuffle(pos_tweets_set)
# random.shuffle(neg_tweets_set)

test_set = pos_tweets_set[:1000] + neg_tweets_set[:10000]
train_set = pos_tweets_set[1000:2000] + neg_tweets_set[1000:2000]

__NaiveBayesClassifier = NaiveBayesClassifier(train_set)
print("Accuracy: {}".format(__NaiveBayesClassifier.accuracy(test_set)))

# save model for later use
pickle.dump(__NaiveBayesClassifier, open("naivebayes.pickle", "wb"))
