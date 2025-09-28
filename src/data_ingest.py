import pandas as pd
from sqlalchemy import create_engine
import os
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_data(csv_path, db_path, table_name):
    """
    Reads taxi trip data from a CSV file, converts datetime columns,
    and ingests it into a SQLite database table.
    """
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))
        logging.info(f"Created directory: {os.path.dirname(db_path)}")

    logging.info(f"Connecting to database at {db_path}...")
    engine = create_engine(f'sqlite:///{db_path}')

    expected_columns = [
        'pickup_datetime', 'dropoff_datetime', 'pickup_longitude',
        'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'trip_duration'
    ]

    chunksize = 100000
    chunk_num = 0

    logging.info(f"Reading and ingesting data from {csv_path} in chunks...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        try:
            # Keep only the desired columns
            chunk = chunk[expected_columns]

            chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'])
            chunk['dropoff_datetime'] = pd.to_datetime(chunk['dropoff_datetime'])

            if chunk_num == 0:
                chunk.to_sql(table_name, engine, index=False, if_exists='replace')
            else:
                chunk.to_sql(table_name, engine, index=False, if_exists='append')

            chunk_num += 1
            logging.info(f"Ingested chunk {chunk_num}...")
        except KeyError as e:
            logging.error(f"Column error in chunk {chunk_num+1}: {e}. Skipping this chunk.")
            continue

    logging.info("Data ingestion complete.")

if __name__ == '__main__':
    CSV_PATH = 'data/uber_data.csv'
    DB_PATH = 'data/uber.db'
    TABLE_NAME = 'uber_trips'

    if os.path.exists(CSV_PATH):
        ingest_data(CSV_PATH, DB_PATH, TABLE_NAME)
    else:
        logging.error(f"Error: Dataset not found at {CSV_PATH}. Please download the dataset first.")
