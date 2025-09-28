import pandas as pd
from prophet import Prophet
import joblib
import os
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A safe function to calculate MAPE to avoid division by zero
def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error, handling zero values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return 0.0 
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def train_and_save_model(data_path, model_output_path):
    """
    Trains a Prophet forecasting model on the hourly demand data, evaluates it,
    and saves the final trained model.
    """
    logging.info(f"Reading hourly demand data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['ds'])

    # Model Evaluation with a train/test split
    logging.info("Evaluating model performance...")
    split_point = int(len(df) * 0.9)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    eval_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    eval_model.fit(train_df)
    
    future_eval = eval_model.make_future_dataframe(periods=len(test_df), freq='h')
    forecast_eval = eval_model.predict(future_eval)
    
    y_true = test_df['y'].values
    y_pred = forecast_eval['yhat'][-len(test_df):].values
    
    # --- Calculate and log all evaluation metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE in two steps for universal compatibility
    mse = mean_squared_error(y_true, y_pred) # 1. Calculate MSE
    rmse = np.sqrt(mse)                      # 2. Take the square root
    
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logging.info("--- Model Evaluation Results ---")
    logging.info(f"Mean Absolute Error (MAE):      {mae:.2f} rides")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f} rides")
    logging.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    logging.info(f"R-squared (R²):                 {r2:.2f}")
    logging.info("---------------------------------")

    # Final Model Training
    logging.info("Training final model on all data...")
    final_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    final_model.fit(df)

    if not os.path.exists(os.path.dirname(model_output_path)):
        os.makedirs(os.path.dirname(model_output_path))
        
    logging.info(f"Saving final model to {model_output_path}...")
    joblib.dump(final_model, model_output_path)
        
    logging.info("Model training and saving complete.")

if __name__ == '__main__':
    HOURLY_DEMAND_PATH = 'data/demand_by_hour.csv'
    MODEL_PATH = 'models/prophet_model.pkl'

    if os.path.exists(HOURLY_DEMAND_PATH):
        train_and_save_model(HOURLY_DEMAND_PATH, MODEL_PATH)
    else:
        logging.error(f"Hourly demand data not found at {HOURLY_DEMAND_PATH}. Please run feature_engineering.py first.")