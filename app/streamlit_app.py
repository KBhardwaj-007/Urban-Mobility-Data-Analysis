import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Urban Mobility Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Loading Data and Model ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned.csv', parse_dates=['pickup_datetime'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load('models/prophet_model.pkl')
    return model

df = load_data()
model = load_model()

# --- Sidebar Filters ---
st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("---")

# Consistent column name for filtering
datetime_col = 'pickup_datetime'

# Date Range Selector
min_date = df[datetime_col].min().date()
max_date = df[datetime_col].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

start_date, end_date = date_range

selected_days = st.sidebar.multiselect(
    "Filter by Day of Week",
    options=df[datetime_col].dt.day_name().unique(),
    default=df[datetime_col].dt.day_name().unique()
)

# --- Filter Dataframe based on all selections ---
filtered_df = df[
    (df[datetime_col].dt.date >= start_date) &
    (df[datetime_col].dt.date <= end_date) &
    (df[datetime_col].dt.day_name().isin(selected_days))
]

# --- Main Dashboard ---
st.title("🚕 Urban Mobility: Ride Demand & Forecasting")
st.markdown("An interactive dashboard for analyzing and forecasting Uber ride demand in NYC.")

# --- Key Metrics (KPIs) ---
st.header("Key Performance Indicators")
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    total_trips = filtered_df.shape[0]
    avg_duration = filtered_df['trip_duration'].mean()
    mode_passengers = filtered_df['passenger_count'].mode()[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trips", f"{total_trips:,}")
    col2.metric("Average Trip Duration (min)", f"{avg_duration:.2f}")
    col3.metric("Most Common Trip Size", f"{mode_passengers} Passenger(s)")

st.markdown("---")

# --- Tabs for different analyses ---
tab1, tab2, tab3 = st.tabs(["Geospatial Hotspots", "Demand Patterns", "Demand Forecast"])

with tab1:
    st.header("📍 Geospatial Demand Hotspots")
    st.markdown("Visualizing trip pickup locations across the city.")
    if filtered_df.empty:
        st.warning("No data to display for the selected filters.")
    else:
        st.warning(
            "⚠️ **Performance Warning:** Displaying a large number of points (>50,000) can slow down or crash the app."
        )
        max_points = filtered_df.shape[0]
        slider_max = min(max_points, 100000)
        num_points_to_plot = st.slider(
            "Select number of trips to visualize:", 1000, slider_max, 5000, 1000
        )
        map_df = filtered_df[['pickup_latitude', 'pickup_longitude']].dropna().sample(num_points_to_plot)
        map_df = map_df.rename(columns={'pickup_latitude': 'latitude', 'pickup_longitude': 'longitude'})
        st.map(map_df)

with tab2:
    st.header("📊 Demand Patterns Analysis")
    if filtered_df.empty:
        st.warning("No data to display for the selected filters.")
    else:
        # 1. Hourly Demand using Plotly
        st.subheader("Hourly Ride Demand")
        hourly_demand = filtered_df.set_index(datetime_col).resample('h').size().reset_index(name='trip_count')
        fig_hourly = px.bar(hourly_demand, x=datetime_col, y='trip_count', title='Total Trips by Hour')
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # 2. Demand Heatmap using Seaborn
        st.subheader("Demand Heatmap: Hour of Day vs. Day of Week")
        heatmap_data = filtered_df.groupby([filtered_df[datetime_col].dt.hour, filtered_df[datetime_col].dt.day_name()]).size().unstack()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(columns=day_order)

        fig_heatmap, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap='hot_r', linecolor='white', linewidths=0.5, ax=ax)
        st.pyplot(fig_heatmap)

        # Business Insights as Text
        st.info(
            """
            **Insight:** The heatmap clearly shows two daily peaks on weekdays: a morning rush (7-9 AM) and an evening peak (5-7 PM). 
            Demand is significantly higher and later on Fridays and Saturdays, indicating strong leisure activity. 
            This information is crucial for optimizing driver allocation and implementing surge pricing.
            """
        )

        # Passenger Count Analysis
        st.markdown("---") # Visual separator
        st.subheader("Passenger Count Analysis")
        
        # Dataframe for the passenger count distribution
        passenger_counts = filtered_df['passenger_count'].value_counts().reset_index()
        passenger_counts.columns = ['Passenger Count', 'Number of Trips']
        
        # Plotly bar chart
        fig_passengers = px.bar(
            passenger_counts,
            x='Passenger Count',
            y='Number of Trips',
            title='Distribution of Trips by Passenger Count',
            labels={'Number of Trips': 'Total Number of Trips'},
            color='Number of Trips',
            color_continuous_scale=px.colors.sequential.Teal
        )
        st.plotly_chart(fig_passengers, use_container_width=True)

        st.info(
            """
            **Insight:** This chart shows the complete breakdown of trip sizes. Overwhelmingly, most trips consist of a single passenger,
            which strongly indicates that a majority of rides are for individual commutes or travel. 
            This data is vital for fleet management and marketing strategies.
            """
        )
        

with tab3:
    st.header("📈 Future Ride Demand Forecast")
    
    future_days = st.slider("Select number of days to forecast:", 7, 90, 30, key="forecast_slider")
    
    future = model.make_future_dataframe(periods=future_days * 24, freq='h')
    forecast = model.predict(future)
    
    st.subheader("Forecasted Hourly Ride Demand")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.subheader("Forecast Components")
    # Business Insights as Text
    st.info(
        """
        **About the Components:** The model breaks down the forecast into three parts:
        - **Trend:** The long-term direction of ride demand. An upward trend suggests overall growth.
        - **Weekly Seasonality:** The typical pattern of demand over a week (e.g., lower on Monday, higher on Friday).
        - **Daily Seasonality:** The typical pattern of demand over a 24-hour period (e.g., morning and evening peaks).
        """
    )
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
