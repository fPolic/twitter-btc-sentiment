
from ml.helpers.mongo import get_tweet_repo, get_db_instance

TweetRepo = get_tweet_repo()

pipelineHourly = [
    {
        '$lookup': {
            'from': 'lexicon',
            'localField': 'tokens',
            'foreignField': 'word',
            'as': 'emotions'
        }
    }, {
        '$unwind': '$emotions'
    }, {
        '$project': {
            '_id': 0,
            'date': {
                '$dateToString': {
                    'date': '$date',
                    'format': '%Y-%m-%dT%HH'
                }
            },
            'anger': '$emotions.anger',
            'anticipation': '$emotions.anticipation',
            'disgust': '$emotions.disgust',
            'fear': '$emotions.fear',
            'joy': '$emotions.joy',
            'negative': '$emotions.negative',
            'positive': '$emotions.positive',
            'sadness': '$emotions.sadness',
            'surprise': '$emotions.surprise',
            'trust': '$emotions.trust'
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
        '$addFields': {
            'date': {
                '$dateFromString': {
                    'dateString': '$_id',
                    'format': '%Y-%m-%dT%HH'
                }
            }
        }
    }, {
        '$sort': {
            'date': 1
        }
    }, {
        '$out': 'emotions_aggregation'
    }
]


def create_emotions_aggregation_view(hourly=False):

    config = {'create': 'emotions_by_houre',
              'viewOn': 'tweets', 'pipeline': pipelineHourly}

    get_db_instance().command(config)


def aggregate_tweet_emotions_by_houre():
    get_db_instance().emotions_aggregation.drop()
    TweetRepo.aggregate(pipelineHourly)
