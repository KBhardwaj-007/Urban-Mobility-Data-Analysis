import pandas as pd
from sqlalchemy import create_engine, text
import os
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(db_path, table_name, output_path):
    """
    Connects to the SQLite database, cleans the trip data, and saves it.
    """
    logging.info(f"Connecting to database: {db_path}")
    engine = create_engine(f'sqlite:///{db_path}')
    
    query = text(f"SELECT * FROM {table_name}")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=['pickup_datetime', 'dropoff_datetime'])

    logging.info(f"Initial data size: {len(df)}")
    logging.info("Cleaning data...")
    
    df.dropna(inplace=True)
    df = df[(df['passenger_count'] > 0)]

    # Geographic Bounding Box Filter
    NYC_BOUNDS = {
        'min_lon': -74.3, 'max_lon': -73.7,
        'min_lat': 40.5, 'max_lat': 41.0,
    }
    
    logging.info("Applying geographical bounding box filter...")
    df = df[
        (df['pickup_latitude'].between(NYC_BOUNDS['min_lat'], NYC_BOUNDS['max_lat'])) &
        (df['pickup_longitude'].between(NYC_BOUNDS['min_lon'], NYC_BOUNDS['max_lon'])) &
        (df['dropoff_latitude'].between(NYC_BOUNDS['min_lat'], NYC_BOUNDS['max_lat'])) &
        (df['dropoff_longitude'].between(NYC_BOUNDS['min_lon'], NYC_BOUNDS['max_lon']))
    ]
    logging.info(f"Data size after geo-filter: {len(df)}")
    
    # Filter trip_duration, which is in seconds (1 min to 1:30 hours)
    logging.info("Applying trip duration filter (in seconds)...")
    df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 5400)]
    
    # 'trip_duration_minutes' column for easier interpretation
    df['trip_duration_minutes'] = round(df['trip_duration'] / 60.0, 2)

    logging.info(f"Final cleaned data size: {len(df)}")
    
    logging.info(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    logging.info("Data cleaning complete.")

if __name__ == '__main__':
    DB_PATH = 'data/uber.db'
    TABLE_NAME = 'uber_trips'
    CLEANED_CSV_PATH = 'data/cleaned.csv'

    if os.path.exists(DB_PATH):
        clean_data(DB_PATH, TABLE_NAME, CLEANED_CSV_PATH)
    else:
        logging.error(f"Database not found at {DB_PATH}. Please run data_ingest.py first.")