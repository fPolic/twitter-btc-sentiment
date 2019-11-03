
from ml.helpers.mongo import get_tweet_repo, get_db_instance

TweetRepo = get_tweet_repo()

pipelineHourly = [
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
                    'format': '%Y-%m-%dT%HH'
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
    }
]

pipelineDaily = [
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
        '$addFields': {
            'date': {
                '$dateFromString': {
                    'dateString': '$_id'
                }
            }
        }
    }, {
        '$sort': {
            'date': 1
        }
    }
]


def create_emotions_aggregation_view(hourly=False):
    config = {'create': 'tweet_emotions_by_date',
              'viewOn': 'tweets', 'pipeline': pipelineDaily}
    if hourly:
        config = {'create': 'tweet_emotions_by_date_hourly',
                  'viewOn': 'tweets', 'pipeline': pipelineHourly}

    get_db_instance().command(config)


def aggregate_tweet_emotions_by_date():
    for c in TweetRepo.aggregate(pipelineHourly):
        print(c)
        print('\n')
