from ml.helpers.mongo import get_tweet_repo, get_db_instance
TweetRepo = get_tweet_repo()

pipeline = [
    {
        '$lookup': {
            'from': 'VAD',
            'localField': 'tokens',
            'foreignField': 'Word',
            'as': 'vad'
        }
    }, {
        '$unwind': '$vad'
    }, {
        '$project': {
            '_id': 0,
            'date': {
                '$dateToString': {
                    'date': '$date',
                    'format': '%Y-%m-%dT%HH'
                }
            },
            'arousal': '$vad.Arousal',
            'dominance': '$vad.Dominance',
            'valence': '$vad.Valence'
        }
    }, {
        '$group': {
            '_id': '$date',
            'arousal': {
                '$sum': '$arousal'
            },
            'dominance': {
                '$sum': '$dominance'
            },
            'valence': {
                '$sum': '$valence'
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
        '$out': 'vad_aggregate'
    }
]


def aggregate_vad():
    get_db_instance().vad_aggregate.drop()
    TweetRepo.aggregate(pipeline)
