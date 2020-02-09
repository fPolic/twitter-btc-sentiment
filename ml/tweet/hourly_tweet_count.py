from ml.helpers.mongo import get_tweet_repo, get_db_instance
TweetRepo = get_tweet_repo()

pipeline = [
    {
        '$project': {
            'date': {
                '$dateToString': {
                    'date': '$date',
                    'format': '%Y-%m-%dT%HH'
                }
            }
        }
    }, {
        '$group': {
            '_id': '$date',
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
        '$out': 'tweetCount'
    }
]


def count_tweets():
    get_db_instance().tweetCount.drop()
    TweetRepo.aggregate(pipeline)
