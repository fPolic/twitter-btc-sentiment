
from ml.helpers.mongo import get_db_instance


pipeline = [
    {
        '$lookup': {
            'from': 'btc',
            'localField': 'date',
            'foreignField': 'date',
            'as': 'btc'
        }
    }, {
        '$lookup': {
            'from': 'spx',
            'localField': 'date',
            'foreignField': 'date',
            'as': 'spx'
        }
    }, {
        '$lookup': {
            'from': 'gld',
            'localField': 'date',
            'foreignField': 'date',
            'as': 'gld'
        }
    }, {
        '$unwind': {
            'path': '$btc',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$unwind': {
            'path': '$spx',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$unwind': {
            'path': '$gld',
            'preserveNullAndEmptyArrays': False
        }
    }, {
        '$project': {
            'btc._id': 0,
            'btc.date': 0,
            'spx._id': 0,
            'spx.date': 0,
            'gld._id': 0,
            'gld.date': 0
        }
    }, {
        '$out': 'time_series'
    }
]


def join_time_series():
    get_db_instance().time_series.drop()
    get_db_instance().emotions_aggregation.aggregate(pipeline)
