#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install boto3 pandas sagemaker --quiet


# In[2]:


import boto3
import pandas as pd
from sagemaker import Session
import os


# In[3]:


# Initialize the SageMaker session
from sagemaker import Session

# Create a session and S3 client
session = Session()
s3 = session.boto_session.client("s3")

# Define your bucket and folder paths
bucket_name = "api-smp-data-il-central-1"
raw_data_folder = "raw/"

# List all files in the raw data folder
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=raw_data_folder)

if "Contents" in response:
    print("Raw data files:")
    for obj in response["Contents"]:
        print(obj["Key"])
else:
    print("No files found in the raw data folder.")


# In[4]:


import boto3
import pandas as pd
import json
from datetime import datetime

# Initialize S3 client
s3 = boto3.client('s3')

# Define S3 bucket and file details
bucket_name = 'api-smp-data-il-central-1'
current_date = datetime.now().strftime('%Y-%m-%d')  # Format the current date as yyyy-mm-dd
raw_file_key = f"raw/{current_date}_data.json"  # Construct the file key dynamically

# Download and process the most recent file
local_file_path = "/tmp/raw_data.json"

try:
    # Download the file from S3
    s3.download_file(Bucket=bucket_name, Key=raw_file_key, Filename=local_file_path)
    print(f"Downloaded {raw_file_key} to {local_file_path}")

    # Load and process the JSON file
    with open(local_file_path, "r") as file:
        raw_data = json.load(file)

    # Flatten the JSON structure
    data_records = []
    for record in raw_data:
        date = record["date"]
        for smp_record in record["smpData"]:
            data_records.append({
                "datetime": f"{date} {smp_record['time']}",
                "price": smp_record["day_Ahead_Constrained_Smp"]
            })

    # Load into a DataFrame
    df = pd.DataFrame(data_records)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M:%S")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Display the first few rows
    print("Processed Data:")
    print(df.head())

except Exception as e:
    print(f"Failed to process the file: {e}")


# In[5]:


# Feature Engineering
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month

# Lagged Features
df['price_lag_1'] = df['price'].shift(1)
df['price_lag_7'] = df['price'].shift(7 * 96)  # Weekly lag
df['price_lag_30'] = df['price'].shift(30 * 96)  # Monthly lag
df['lag_1_year'] = df['price'].shift(365 * 96)  # 1-year lag
df['lag_2_years'] = df['price'].shift(2 * 365 * 96)  # 2-year lag

# Rolling Features
df['rolling_avg_7'] = df['price'].rolling(window=7 * 96).mean()
df['rolling_avg_30'] = df['price'].rolling(window=30 * 96).mean()

# Yearly/Seasonal Features
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Handle missing values
df.fillna(method='bfill', inplace=True)

# Verify dataset
print("Dataset overview after feature engineering:")
print(df.head())

# Historical Testing Period
historical_test_start = '2023-01-01'
historical_test_end = '2024-06-01'

train_data = df[df['datetime'] < historical_test_start]
test_data = df[(df['datetime'] >= historical_test_start) & (df['datetime'] <= historical_test_end)]

print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")


# In[6]:


from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Define historical testing period
historical_test_start = pd.Timestamp('2023-01-01')
historical_test_end = pd.Timestamp('2024-12-01')

# Ensure datetime is in correct format
df['datetime'] = pd.to_datetime(df['datetime'])

# Debug: Print dataset date range
print(f"Dataset date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Split into training and testing sets based on dates
train_data = df[df['datetime'] < historical_test_start]
test_data = df[(df['datetime'] >= historical_test_start) & (df['datetime'] <= historical_test_end)]

# Debug: Print the sizes of train and test data
print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")

# Ensure train_data and test_data are not empty
if train_data.empty or test_data.empty:
    raise ValueError("Insufficient data for the specified historical testing period.")

# Select features and target
features = ['hour', 'day_of_week', 'month', 'price_lag_1', 'price_lag_7', 'price_lag_30', 
            'rolling_avg_7', 'rolling_avg_30', 'is_weekend', 'lag_1_year']

X_train = train_data[features]
y_train = train_data['price']

X_test = test_data[features]
y_test = test_data['price']

# Ensure X_train and y_train are not empty
if X_train.empty or y_train.empty:
    raise ValueError("Training data is empty after filtering.")

# Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Define 6-month intervals
intervals = pd.date_range(start=historical_test_start, end=historical_test_end, freq='6M')
intervals = list(intervals) + [historical_test_end]  # Ensure the last date is included

# Calculate MAE for each interval
mae_scores = []
for i in range(len(intervals) - 1):
    start = intervals[i]
    end = intervals[i + 1]

    # Filter test data for the current interval
    interval_data = test_data[(test_data['datetime'] >= start) & (test_data['datetime'] < end)]
    if interval_data.empty:
        print(f"No data available for interval {start} to {end}")
        continue

    # Predict for the interval
    X_interval = interval_data[features]
    y_interval = interval_data['price']
    y_interval_pred = model.predict(X_interval)

    # Calculate MAE
    interval_mae = mean_absolute_error(y_interval, y_interval_pred)
    mae_scores.append((start, end, interval_mae))
    print(f"MAE for interval {start.date()} to {end.date()}: {interval_mae}")

# Visualize the predictions for the entire period
y_pred = model.predict(X_test)  # Predict for the entire test set

plt.figure(figsize=(12, 6))
plt.plot(test_data['datetime'], y_test, label='Actual Prices', color='blue', alpha=0.6)
plt.plot(test_data['datetime'], y_pred, label='Predicted Prices', color='orange', alpha=0.8)
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.title('Historical Prediction vs. Actual')
plt.legend()
plt.grid(True)
plt.show()

# Print MAE results for intervals
print("\nMAE for 6-Month Intervals:")
for start, end, mae in mae_scores:
    print(f"From {start.date()} to {end.date()}: MAE = {mae}")


# In[ ]:


# Your existing code
import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Generate future dates
current_date = pd.Timestamp.now()
target_month = (current_date.month + 2) % 12 or 12
target_year = current_date.year + (1 if current_date.month + 2 > 12 else 0)
_, last_day = calendar.monthrange(target_year, target_month)

future_dates = pd.date_range(
    start=f"{target_year}-{target_month:02d}-01",
    end=f"{target_year}-{target_month:02d}-{last_day}",
    freq='15T'  # Adjust to match the frequency of your data
)

# Create features for future dates
future_features = pd.DataFrame({
    'datetime': future_dates
})
future_features['hour'] = future_features['datetime'].dt.hour
future_features['day_of_week'] = future_features['datetime'].dt.dayofweek
future_features['month'] = future_features['datetime'].dt.month

# Add lagged features based on the last known data from the training set
last_known_data = df.iloc[-1]  # Get the last row of the dataset
future_features['price_lag_1'] = last_known_data['price']
future_features['price_lag_7'] = last_known_data['price_lag_7']
future_features['price_lag_30'] = last_known_data['price_lag_30']
future_features['lag_1_year'] = last_known_data['lag_1_year']

# Use the rolling averages from the most recent data
future_features['rolling_avg_7'] = last_known_data['rolling_avg_7']
future_features['rolling_avg_30'] = last_known_data['rolling_avg_30']

# Add the is_weekend feature
future_features['is_weekend'] = future_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Reorder to match the feature order used during training
future_features = future_features[['hour', 'day_of_week', 'month', 'price_lag_1', 'price_lag_7', 
                                   'price_lag_30', 'rolling_avg_7', 'rolling_avg_30', 
                                   'is_weekend', 'lag_1_year']]

# Predict future prices
future_predictions = model.predict(future_features)
future_df = pd.DataFrame({
    'datetime': future_dates,
    'predicted_price': future_predictions
})

# Display predictions
print("Future price predictions:")
print(future_df.head())

# Plot future predictions alongside previous 2 years' pricing
plt.figure(figsize=(12, 8))

# Plot predicted future prices
plt.plot(future_df['datetime'], future_df['predicted_price'], label='Predicted Prices (Future)', color='orange')

# Add previous 2 years' data for the same period
for year_offset in [1, 2]:  # Include the last 2 years
    year_start = future_dates[0] - pd.DateOffset(years=year_offset)
    year_end = future_dates[-1] - pd.DateOffset(years=year_offset)
    historical_data = df[(df['datetime'] >= year_start) & (df['datetime'] <= year_end)]
    if not historical_data.empty:
        plt.plot(historical_data['datetime'] + pd.DateOffset(years=year_offset), 
                 historical_data['price'], 
                 label=f'Actual Prices ({future_dates[0].year - year_offset})', linestyle='dashed')

# Add labels and legend
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.title('Future Price Predictions with Historical Comparison')
plt.legend()
plt.grid(True)

# Show the graph
plt.show()


# In[ ]:


from datetime import datetime

# Save to a CSV file
current_date_str = datetime.now().strftime("%Y-%m-%d")  # Get current date in YYYY-MM-DD format
output_file = f"/tmp/future_predictions_{current_date_str}.csv"  # Add date to the filename
future_df.to_csv(output_file, index=False)

# Upload to S3
output_bucket = "api-smp-data-il-central-1"
output_key = f"processed/future_predictions_{current_date_str}.csv"  # Add date to S3 key

s3.upload_file(output_file, output_bucket, output_key)
print(f"Predictions saved to S3: s3://{output_bucket}/{output_key}")


# In[ ]:




