from ml.helpers.mongo import get_db_instance

pipeline = [
    {
        '$group': {
            '_id': '$username',
            'count': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            'count': -1
        }
    }, {
        '$out': 'user_post_count'
    }
]


def generate_user_post_counts():
    get_db_instance().user_post_count.drop()
    get_db_instance().tweets.aggregate(pipeline, allowDiskUse=True, cursor={})
