import dash
from dash import dcc, html
import dash.dependencies as dd
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask

# Create a Flask server
server = Flask(__name__)

# Create a Dash app
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

@app.callback(
    dash.dependencies.Output('forecast-graph', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Load data and perform forecasting
    data = pd.read_excel("C:/Users/sneha/OneDrive/Desktop/train scheduling/passenger flow/04.02.2024_AFC-KMRL-Data_-_1400_Hrs.xlsx")
    
    # Convert Date and Time to a single datetime column
    data['Datetime'] = pd.to_datetime(data['Transaction Time'])
    data = data[data['Fare Product'] != 'Staff Card']
    data = data.drop(['Equipment Type', 'Equipment ID', 'Fare Media', 'Fare', 'Fare Product'], axis=1)

    # Group data by hour and station to get passenger flow counts
    data['hour'] = data['Datetime'].dt.floor('h')
    hourly_passenger_flow = data.groupby(['hour', 'Station', 'Date']).size().reset_index(name='Passenger Flow')

    # Pivot the data to have hours as index and stations as columns
    pivot_data = hourly_passenger_flow.pivot(index='hour', columns='Station', values='Passenger Flow').fillna(0)

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

    # Create a DataFrame for the forecasted values
    forecast_df = pd.DataFrame(forecast_values, index=forecast_hours)

    # Create a Plotly figure
    fig = go.Figure()
    
    # Add traces for historical data and forecast
    for station in pivot_data.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot_data.index,
                y=pivot_data[station],
                mode='lines',
                name=f'Historical {station}',
                visible=True  # Make historical data visible
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_hours,
                y=forecast_values[station],
                mode='lines',
                name=f'Forecast {station}',
                visible=False  # Make forecast initially invisible
            )
        )

    # Create dropdown options to toggle each station's visibility
    dropdown_buttons = [
        dict(
            label=station,
            method="update",
            args=[{"visible": [True if 'Historical' in trace.name or station in trace.name else False for trace in fig.data]},
                  {"title": f"Passenger Count Over Time - {station}"}]
        )
        for station in pivot_data.columns
    ]
    # Update layout with dropdown menu
    fig.update_layout(
        title="Passenger Count Over Time - Select a Station",
        xaxis_title="Time",
        yaxis_title="Passenger Count",
        updatemenus=[{
            "buttons": dropdown_buttons,
            "direction": "down",
            "showactive": True,
        }]
    )

    return fig

# Define the layout of the app
app.layout = html.Div([
    html.H1("SARIMA Forecasting Dashboard"),
    dcc.Graph(id='forecast-graph'),
    dcc.Interval(id='interval-component', interval=1*3600*1000)  # Update every hour (1 hour = 3600 seconds)
])

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)