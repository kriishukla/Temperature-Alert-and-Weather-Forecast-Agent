import requests
from uagents import Agent, Context, Bureau, Model
from uagents.setup import fund_agent_if_low


import requests
from datetime import datetime, timedelta


from agents.temperature.TemperatureAlertAgent import alert_agent

from agents.temperature.message_printer import message_printer

def get_weather_forecast(location, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"  \
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch weather data. Error: {response.status_code}")
        return None

def save_weather_forecast_to_file(data, file_name):
    if data is None:
        return

    with open(file_name, 'w') as file:
        file.write(f"Weather forecast for {data['city']['name']} for the next 7 days:\n")
        for entry in data['list']:
            date_time = datetime.fromtimestamp(entry['dt'])
            temperature = entry['main']['temp']
            description = entry['weather'][0]['description']

            file.write(f"{date_time.strftime('%Y-%m-%d %H:%M:%S')} - Temperature: {temperature}°C, Description: {description}\n")

if __name__ == "__main__":
    bureau = Bureau()
    bureau.add(alert_agent)
    bureau.add(message_printer)

    bureau.run()
    location = input("Enter Location (if you want to get weather forecast for this week else enter 99): ")
    if location=="99":
        exit()

    api_key = "64265fd3eb57b27cd8822fb8923982b0"  
    file_name = "prediction.txt"

    weather_data = get_weather_forecast(location, api_key)

    if weather_data:
        save_weather_forecast_to_file(weather_data, file_name)
        print(f"Weather forecast saved to {file_name}")
