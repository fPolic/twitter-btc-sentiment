import os
import csv

from datetime import datetime
from pymongo import MongoClient

DIR = './ml/app/timeseries'
DATABASE_NAME = 'fer'
mongoURL = 'mongodb://localhost:27017/'


def load_symbol_data(symbol):
    series = []
    file_path = os.path.join(DIR, symbol + '.csv')

    with open(file_path) as csv_file:
        for line in csv.DictReader(csv_file):

            format = '%Y-%m-%d %H:%M:%S'
            parsed_date = datetime.strptime(line.get('date'), format)
            series.append({
                "date": parsed_date,
                "low": float(line.get("low")),
                "open": float(line.get("open")),
                "high": float(line.get("high")),
                "close": float(line.get("close")),
                "volume_btc": float(line.get("volume_btc")),
                "volume_usd": float(line.get("volume_usd") if line.get("volume_usd") else line.get("volume_usdt")),
            })

    return series


def main():
    client = MongoClient(mongoURL)
    db = client[DATABASE_NAME]

    db.binance.drop()
    db.coinbase.drop()
    db.bitfinex.drop()

    db.binance.insert_many(load_symbol_data('binance'))
    db.bitfinex.insert_many(load_symbol_data('bitfinex'))
    db.coinbase.insert_many(load_symbol_data('coinbase'))


if __name__ == "__main__":
    main()
