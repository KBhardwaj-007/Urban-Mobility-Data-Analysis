import pandas as pd
import os
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_hourly_demand(cleaned_data_path, output_path):
    """
    Reads the cleaned trip data and aggregates it to create a time-series
    of hourly ride demand, formatted for Prophet.
    """
    logging.info(f"Reading cleaned data from {cleaned_data_path}...")
    df = pd.read_csv(cleaned_data_path, parse_dates=['pickup_datetime'])
    df.set_index('pickup_datetime', inplace=True)

    logging.info("Aggregating data by the hour...")
    demand_by_hour = df.resample('h').size().asfreq('h', fill_value=0).reset_index(name='demand')
    demand_by_hour.rename(columns={'pickup_datetime': 'ds', 'demand': 'y'}, inplace=True)

    logging.info(f"Saving hourly demand data to {output_path}...")
    demand_by_hour.to_csv(output_path, index=False)
    logging.info("Hourly demand dataset created successfully.")

if __name__ == '__main__':
    CLEANED_CSV_PATH = 'data/cleaned.csv'
    HOURLY_DEMAND_PATH = 'data/demand_by_hour.csv'

    if os.path.exists(CLEANED_CSV_PATH):
        create_hourly_demand(CLEANED_CSV_PATH, HOURLY_DEMAND_PATH)
    else:
        logging.error(f"Cleaned data not found at {CLEANED_CSV_PATH}. Please run data_clean.py first.")