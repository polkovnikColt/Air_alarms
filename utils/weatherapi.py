import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
API_KEY = os.getenv("WEATHER_API_KEY")


def get_weather_json(location, date_str):
    url = f"{BASE_URL}/{location}/{date_str}?unitGroup=metric"
    print(url)
    url += f"&key={API_KEY}"

    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, headers=headers, data={})
    response.raise_for_status()

    if response.status_code != 200:
        raise Exception(f"Error getting weather: {response.status_code}")

    json_data = response.json()
    return json_data


def get_hourly_weather_json(location, date_str):
    json_data = get_weather_json(location, date_str)
    result = []

    city_data = {
        "city_latitude": json_data["latitude"],
        "city_longitude": json_data["longitude"],
        "city_resolvedAddress": json_data["resolvedAddress"],
        "city_address": json_data["address"],
        "city_timezone": json_data["timezone"],
        "city_tzoffset": json_data["tzoffset"],
    }

    day_data = { "day_severerisk": 10 }
    for attribute, value in json_data["days"][0].items():
        if (attribute != "hours"):
            day_data[f"day_{attribute}"] = value

    for hour in json_data["days"][0]["hours"]:
        hour_data = { "hour_severerisk": 10 }
        hour_data.update(city_data)
        hour_data.update(day_data)

        for attribute, value in hour.items():
            hour_data[f"hour_{attribute}"] = value

        result.append(hour_data)

    return result
