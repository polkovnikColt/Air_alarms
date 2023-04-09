import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

# Define endpoint parameters
BASE_URL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline'
API_KEY = os.getenv('WEATHER_API_KEY')

def get_weather_json(location, date_str):
	ts = datetime.strptime(date_str, "%Y-%m-%d")
	ts = int(ts.replace(minute=0, second=0, microsecond=0).utcnow().timestamp())
	ts_plus_12hours = ts + 3600 * 12
	# Construct API URL
	url = f"{BASE_URL}/{location}/{ts}/{ts_plus_12hours}?key={API_KEY}"
	print(url)

	# Send GET request to API endpoint
	# response = requests.get(url)
	headers = {
			'Content-Type': "application/json"
			}
	response = requests.request("GET", url, headers=headers, data={})
	response.raise_for_status()
	json_data = response.json()
	weather = json_data["days"]
	for i in weather:
			weatherHours = i["hours"]
			json_string = json.dumps(weatherHours)
			data = json.loads(json_string)

	# filter the list using a condition
			filtered_list = [d for d in data if d["datetimeEpoch"] >= ts and d['datetimeEpoch'] <= ts_plus_12hours]

	# convert the filtered list back to JSON
			json_result = json.dumps(filtered_list)
			i["hours"] = json_result
	json_data["days"] = weather

	# Check response status code
	if response.status_code == 200:
			# Print response content
			return json_data
	else:
			# Print error message
			print(f"Error: {response.status_code}")