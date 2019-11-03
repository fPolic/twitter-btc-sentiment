
from ml.helpers.mongo import get_tweet_repo, get_db_instance

TweetRepo = get_tweet_repo()

pipeline = [
    # {'$sample': {'size': 150}},
    {
        '$sort': {
            'date': 1
        }
    }, {
        '$project': {
            '_id': 0,
            'date': {
                '$dateToString': {
                    'date': '$date',
                    'format': '%Y-%m-%d'
                }
            },
            'anger': 1,
            'anticipation': 1,
            'disgust': 1,
            'fear': 1,
            'joy': 1,
            'negative': 1,
            'positive': 1,
            'sadness': 1,
            'surprise': 1,
            'trust': 1
        }
    }, {
        '$group': {
            '_id': '$date',
            'anger': {
                '$sum': '$anger'
            },
            'anticipation': {
                '$sum': '$anticipation'
            },
            'disgust': {
                '$sum': '$disgust'
            },
            'fear': {
                '$sum': '$fear'
            },
            'joy': {
                '$sum': '$joy'
            },
            'negative': {
                '$sum': '$negative'
            },
            'positive': {
                '$sum': '$positive'
            },
            'sadness': {
                '$sum': '$sadness'
            },
            'surprise': {
                '$sum': '$surprise'
            },
            'trust': {
                '$sum': '$trust'
            },
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            '_id': 1
        }
    }
]


def create_emotions_aggregation_view():
    get_db_instance().command(
        {'create': 'tweet_emotions_by_date', 'viewOn': 'tweets', 'pipeline': pipeline})


def aggregate_tweet_emotions_by_date():
    for c in TweetRepo.aggregate(pipeline):
        print(c)
        print('\n')
