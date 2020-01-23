from ml.helpers.mongo import get_db_instance

pipeline = [
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
            'word': '$emotions.word',
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
            '_id': '$word',
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
            'count': -1
        }
    }, {
        '$out': 'word_count'
    }
]


def generate_word_count():
    get_db_instance().word_count.drop()
    get_db_instance().tweets.aggregate(pipeline)
