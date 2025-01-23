# app.py
from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.io as pio

    #data input
    data=pd.read_excel("C:/Users/sneha/OneDrive/Desktop/train scheduling/passenger flow/04.02.2024_AFC-KMRL-Data_-_1400_Hrs.xlsx")
    df=data
    data.head()
    df = pd.DataFrame(data)

    # Convert Date and Time to a single datetime column
    data['Datetime'] = pd.to_datetime(data['Transaction Time'])
    data = data[data['Fare Product'] != 'Staff Card']
    data = data.drop(['Equipment Type','Equipment ID','Fare Media','Fare','Fare Product'],axis=1)
    # Group data by hour and station to get passenger flow counts
    data['hour'] = data['Datetime'].dt.floor('h')
    hourly_passenger_flow = data.groupby(['hour', 'Station','Date']).size().reset_index(name='Passenger Flow')
    # Pivot the data to have hours as index and stations as columns
    pivot_data = hourly_passenger_flow.pivot(index='hour', columns='Station', values='Passenger Flow').fillna(0)
    pivot_data
    # Calculate total passenger count for each station
    total_passengers_per_station = pivot_data.sum()

    """# Plot bar chart
    plt.figure(figsize=(14, 6))
    total_passengers_per_station.plot(kind='bar', color='skyblue')
    plt.title("Total Passenger Count per Station")
    plt.xlabel("Station")
    plt.ylabel("Total Passenger Count")
    plt.xticks(rotation=45)
    plt.show()"""

    # Perform SARIMA forecasting logic here
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Create a list of the forecast times for the next day from 7 AM to 10 PM
    forecast_hours = pd.date_range(start='2024-02-04 07:00:00', end='2024-02-04 22:00:00', freq='H')

# Forecast for each station
    forecast_values = {}

    for station in pivot_data.columns:
    # Prepare the time series for each station
      time_series = pivot_data[station]

    # Fit the SARIMA model (parameters can be adjusted)
      model = SARIMAX(time_series,
                    order=(1, 1, 1),   # ARIMA(p, d, q)
                    seasonal_order=(1, 1, 1, 24),  # SARIMA(P, D, Q, s) - s=24 for daily seasonality
                    enforce_stationarity=False,
                    enforce_invertibility=False)
      results = model.fit(disp=False)

    # Forecast the next 16 hours (from 7 AM to 10 PM)
      forecast = results.get_forecast(steps=len(forecast_hours))

    # Store the predicted values for the station
      forecast_values[station] = forecast.predicted_mean

    """# Plotting historical data and forecasted values
      plt.figure(figsize=(10, 6))
      plt.plot(time_series.index, time_series, label='Historical Data', color='blue')
      plt.plot(forecast_hours, forecast.predicted_mean, label='Forecast', color='red')

    # Format the x-axis to show dates clearly
      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
      plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show every 2 hours on x-axis
      plt.gcf().autofmt_xdate()  # Rotate date labels for readability

    # Adding labels and title
      plt.title(f'Forecast for Station: {station}')
      plt.xlabel('Date & Time')
      plt.ylabel('Passenger Count')
      plt.legend()

    # Display the plot
      plt.show()"""

    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Create a list of the forecast times for the next day from 7 AM to 10 PM
    forecast_hours = pd.date_range(start='2024-02-04 07:00:00', end='2024-02-04 22:00:00', freq='H')

    # Initialize a dictionary to store the forecasted values for all stations
    forecast_values = {}

    for station in pivot_data.columns:
       # Prepare the time series for each station
       time_series = pivot_data[station]

       # Fit the SARIMA model (parameters can be adjusted)
       model = SARIMAX(time_series,
                    order=(1, 1, 1),   # ARIMA(p, d, q)
                    seasonal_order=(1, 1, 1, 24),  # SARIMA(P, D, Q, s) - s=24 for daily seasonality
                    enforce_stationarity=False,
                    enforce_invertibility=False)
       results = model.fit(disp=False)

       # Forecast the next 16 hours (from 7 AM to 10 PM)
       forecast = results.get_forecast(steps=len(forecast_hours))


       # Store the predicted values for the station

       forecast_values[station] = forecast.predicted_mean

    # Plotting historical data and forecasted values
       plt.figure(figsize=(10, 6))
       plt.plot(time_series.index, time_series, label='Historical Data', color='blue')
       plt.plot(forecast_hours, forecast.predicted_mean, label='Forecast', color='red')

       # Format the x-axis to show dates clearly
       plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
       plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show every 2 hours on x-axis
       plt.gcf().autofmt_xdate()  # Rotate date labels for readability

    # Adding labels and title
       plt.title(f'Forecast for Station: {station}')
       plt.xlabel('Date & Time')
       plt.ylabel('Passenger Count')
       plt.legend()

    # Display the plot
       plt.show()

   # Create a DataFrame for the forecasted values, using the forecast hours as index
    forecast_df = pd.DataFrame(forecast_values, index=[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,32])
    new_index = pd.date_range(start='2024-02-04 07:00:00', end='2024-02-04 22:00:00', periods=len(forecast_df))
   # Replace the existing index with the new DatetimeIndex
    forecast_df.index = new_index
   # Display the DataFrame with the new datetime index
    print(type(forecast_df.index)) 
   # Export the forecast DataFrame to Excel
    forecast_df.to_excel("forecasted_data.xlsx", sheet_name='Forecast', index=True)

    
    # Generate plot and encode it to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run_server(debug=True)
