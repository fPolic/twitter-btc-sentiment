import os
import csv

from datetime import datetime
from pymongo import MongoClient

DIR = './data/quotes'
DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'


def load_symbol_data(symbol):
    series = []
    file_path = os.path.join(DIR, symbol + '.csv')

    with open(file_path) as csv_file:
        for line in csv.DictReader(csv_file):

            format = '%m/%d/%Y %H:%M'
            date = line.get('Date') + ' ' + line.get('Time')
            parsed_date = datetime.strptime(date, format)

            series.append({
                "date": parsed_date,
                "low": float(line.get("Low")),
                "open": float(line.get("Open")),
                "high": float(line.get("High")),
                "close": float(line.get("Close")),
                "volume": int(line.get("Vol")),
            })

    return series


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]

    db.spx.drop()
    db.gld.drop()
    db.btc.drop()

    db.spx.insert_many(load_symbol_data('SPX'))
    db.gld.insert_many(load_symbol_data('GLD'))
    db.btc.insert_many(load_symbol_data('BTC-USD'))


if __name__ == "__main__":
    main()
