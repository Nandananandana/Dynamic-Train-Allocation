
import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# LOAD AND COMBINE DATA
folder_path = "C:/Users/Lenovo/OneDrive/Desktop/February AFC Data"
file_paths = glob.glob(f'{folder_path}/*.xlsx')
data_frames = [pd.read_excel(file) for file in file_paths]
combined_df = pd.concat(data_frames, ignore_index=True)

# CONVERT DATE AND TIME COLUMNS
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df['Transaction Time'] = pd.to_datetime(combined_df['Transaction Time'].astype(str))

# EXTRACT HOUR AND STATION
combined_df['Hour'] = combined_df['Transaction Time'].dt.hour

# ADDITIONAL FEATURES: DAY OF WEEK AND WEEKEND INDICATOR
combined_df['Day_of_Week'] = combined_df['Date'].dt.dayofweek  # Monday=0, Sunday=6
combined_df['Is_Weekend'] = combined_df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)

# GROUP DATA BY DATE, STATION, HOUR, AND ADDITIONAL FEATURES
hourly_crowd = combined_df.groupby(['Date', 'Station', 'Hour', 'Day_of_Week', 'Is_Weekend']).size().reset_index(name='Crowd')

# NORMALIZATION
scaler = MinMaxScaler(feature_range=(0, 1))
hourly_crowd['Crowd'] = scaler.fit_transform(hourly_crowd[['Crowd']])

# PREPARE SEQUENCES FOR LSTM WITH ADDITIONAL FEATURES
timesteps = 24  # Use 24 hours to predict the next hour
X, y = [], []

for i in range(timesteps, len(hourly_crowd)):
    features = hourly_crowd[['Crowd', 'Day_of_Week', 'Is_Weekend']][i-timesteps:i].values  # Include additional features
    X.append(features)
    y.append(hourly_crowd['Crowd'][i])
    
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # Adjust shape for LSTM input

# BUILD LSTM MODEL
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# TRAIN THE MODEL
model.fit(X, y, epochs=50, batch_size=32)

# PREDICTION FOR MARCH
march_predictions = []
stations = combined_df['Station'].unique()
for station in stations:
    station_predictions = []
    current_input = hourly_crowd[hourly_crowd['Station'] == station]['Crowd'][-timesteps:].values
    
    # Predicting for 31 days (March) with 15 hours per day (8 AM to 10 PM)
    for _ in range(31 * 15):
        current_input_reshaped = current_input.reshape((1, timesteps, X.shape[2]))
        prediction = model.predict(current_input_reshaped)
        station_predictions.append(prediction[0][0])
        current_input = np.append(current_input[1:], prediction[0][0])  # Slide window

    # Inverse transform to original scale
    station_predictions = scaler.inverse_transform(np.array(station_predictions).reshape(-1, 1)).flatten()
    march_predictions.extend([(station, pred) for pred in station_predictions])

# ORGANIZE PREDICTIONS INTO A DATAFRAME
dates = pd.date_range(start='2024-03-01', periods=31, freq='D')
hours = list(range(8, 23))  # 8 AM to 10 PM
hour_labels = [f"{hour}:00" for hour in hours]

# Expand data to fit the format
predicted_df = pd.DataFrame({
    'Station': np.repeat(stations, len(dates) * len(hours)),
    'Date': np.tile(np.repeat(dates, len(hours)), len(stations)),
    'Hour': np.tile(hour_labels, len(dates) * len(stations)),
    'Predicted Crowd': [pred[1] for pred in march_predictions]
})

# SAVE TO EXCEL FILE
output_path = "C:/Users/Lenovo/OneDrive/Desktop/march_predictions_ordered.xlsx"
predicted_df.to_excel(output_path, index=False)

print(f"Predicted crowd levels saved to {output_path}")
