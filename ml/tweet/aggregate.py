
from ml.helpers.mongo import get_tweet_repo, get_db_instance

TweetRepo = get_tweet_repo()

pipelineHourly = [
    {
        '$sort': {
            'date': 1
        }
    }, {
        '$lookup': {
            'from': 'lexicon',
            'localField': 'tokens',
            'foreignField': 'word',
            'as': 'emotions'
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
            'text': 1,
            'emotions': {
                'word': 1,
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
        }
    }, {
        '$unwind': {
            'path': '$emotions',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$group': {
            '_id': '$date',
            'anger': {
                '$sum': '$emotions.anger'
            },
            'anticipation': {
                '$sum': '$emotions.anticipation'
            },
            'disgust': {
                '$sum': '$emotions.disgust'
            },
            'fear': {
                '$sum': '$emotions.fear'
            },
            'joy': {
                '$sum': '$emotions.joy'
            },
            'negative': {
                '$sum': '$emotions.negative'
            },
            'positive': {
                '$sum': '$emotions.positive'
            },
            'sadness': {
                '$sum': '$emotions.sadness'
            },
            'surprise': {
                '$sum': '$emotions.surprise'
            },
            'trust': {
                '$sum': '$emotions.trust'
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
        '$out': 'emotions_hourly'
    }
]

pipelineDaily = [
    {
        '$lookup': {
            'from': 'lexicon',
            'localField': 'tokens',
            'foreignField': 'word',
            'as': 'emotions'
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
            'emotions': {
                'word': 1,
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
        }
    }, {
        '$unwind': {
            'path': '$emotions',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$group': {
            '_id': '$date',
            'anger': {
                '$sum': '$emotions.anger'
            },
            'anticipation': {
                '$sum': '$emotions.anticipation'
            },
            'disgust': {
                '$sum': '$emotions.disgust'
            },
            'fear': {
                '$sum': '$emotions.fear'
            },
            'joy': {
                '$sum': '$emotions.joy'
            },
            'negative': {
                '$sum': '$emotions.negative'
            },
            'positive': {
                '$sum': '$emotions.positive'
            },
            'sadness': {
                '$sum': '$emotions.sadness'
            },
            'surprise': {
                '$sum': '$emotions.surprise'
            },
            'trust': {
                '$sum': '$emotions.trust'
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
    },  {
        '$sort': {
            'date': 1
        },

    },  {
        '$out': 'emotions_daily'
    }
]


def create_emotions_aggregation_view(hourly=False):
    # config = {'create': 'tweet_emotions_by_date',
    #           'viewOn': 'tweets', 'pipeline': pipelineDaily}
    if hourly:
        # config = {'create': 'tweet_emotions_by_date_hourly',
        #           'viewOn': 'tweets', 'pipeline': pipelineHourly}
        TweetRepo.aggregate(pipelineHourly)
    else:
        TweetRepo.aggregate(pipelineDaily)

    # get_db_instance().command(config)


def aggregate_tweet_emotions_by_date():
    for c in TweetRepo.aggregate(pipelineHourly):
        print(c)
        print('\n')
