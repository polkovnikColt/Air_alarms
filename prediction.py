from datetime import date, datetime, timedelta
import pandas as pd
import requests
import scipy.sparse as sp
import json
import os
import pickle

from bs4 import BeautifulSoup

from utils import weatherapi
from utils import isw_preprocessing
from utils import text_preprocessing

MODELS_FOLDER = "data/models"
OUTPUT_FOLDER = "data/prediction"

if not os.path.exists(OUTPUT_FOLDER):
   os.makedirs(OUTPUT_FOLDER)

tfidf_transformer_model = "tfidf_transformer"
tfidf_transformer_version = "v1"

count_vectorizer_model = "count_vectorizer"
count_vectorizer_version = "v1"

logistic_regression_model = "LR"
multi_layer_perceptron_model = "MLP"
random_forest_classifier_model = "RFC"

tfidf = pickle.load(open(f"data/isw/{tfidf_transformer_model}_{tfidf_transformer_version}.pkl", "rb"))
cv = pickle.load(open(f"data/isw/{count_vectorizer_model}_{count_vectorizer_version}.pkl", "rb"))

label_encoder = pickle.load(open(f"data/isw/hour_conditions_label_encoder.pkl", "rb"))

LR = pickle.load(open(f"{MODELS_FOLDER}/{logistic_regression_model}.pkl", "rb"))
MLP = pickle.load(open(f"{MODELS_FOLDER}/{multi_layer_perceptron_model}.pkl", "rb"))
RFC = pickle.load(open(f"{MODELS_FOLDER}/{random_forest_classifier_model}.pkl", "rb"))

REGIONS_DICTIONARY_FILE = "data/data_before_lab_3/regions.csv"
df_regions = pd.read_csv(REGIONS_DICTIONARY_FILE)

weather_exclude = [
	"day_feelslikemax",
	"day_feelslikemin",
	"day_sunriseEpoch",
	"day_sunsetEpoch",
	"day_description",
	"city_latitude",
	"city_longitude",
	"city_address",
	"city_timezone",
	"city_tzoffset",
	"day_feelslike",
	"day_precipprob",
	"day_snow",
	"day_snowdepth",
	"day_windgust",
	"day_windspeed",
	"day_winddir",
	"day_pressure",
	"day_cloudcover",
	"day_visibility",
	"day_severerisk",
	"day_conditions",
	"day_icon",
	"day_source",
	"day_preciptype",
	"day_stations",
	"hour_icon",
	"hour_source",
	"hour_stations",
	"hour_feelslike",
	"hour_sunrise",
	"hour_sunset",
	"hour_sunriseEpoch",
	"hour_sunsetEpoch",
	"hour_moonphase"
]

fields_to_exclude = [
	"city_resolvedAddress",
	"day_datetime",
	"day_datetimeEpoch",
	"hour_datetime",
	"hour_datetimeEpoch",
	"city",
	"region",
	"center_city_ua",
	"center_city_en"
]

tmp_fields_to_exclude = [
	"day_sunrise",
	"day_sunset",
	"hour_preciptype",
	"hour_conditions",
	"hour_solarenergy",
	"region_alt"
]


def get_weather(region, date_str, debug = False):
	city = df_regions[df_regions["region_alt"]==region]["center_city_en"].values[0]
	location = f"{city},Ukraine"

	weather_forecast_dir = f"{OUTPUT_FOLDER}/weather/{city.lower()}"
	weather_forecast_file_name = f"weather___{city.lower()}__{date_str}.json"

	if not os.path.isfile(f"{weather_forecast_dir}/{weather_forecast_file_name}"):
		city_weather_json = weatherapi.get_hourly_weather_json(location, date_str)
		json_object = json.dumps(city_weather_json, indent=4)

		if not os.path.exists(weather_forecast_dir):
			os.makedirs(weather_forecast_dir)
		
		with open(f"{weather_forecast_dir}/{weather_forecast_file_name}", "w") as outfile:
			if debug:
					print(f"Writing {weather_forecast_file_name}")
			outfile.write(json_object)
	elif debug:
		print(f"Weather data for {region} and {date_str} is downloaded. Reading...")
	
	weather_for_day_hourly = json.load(open(f"{weather_forecast_dir}/{weather_forecast_file_name}", "rb"))
	weather_df = pd.DataFrame.from_dict(weather_for_day_hourly)

	weather_df["day_datetime"] = pd.to_datetime(weather_df["day_datetime"])
	weather_df = weather_df.fillna(-10)
	weather_df["city"] = weather_df["city_resolvedAddress"].apply(lambda x: x.split(",")[0])
	weather_df["city"] = weather_df["city"].replace("Хмельницька область", "Хмельницький")

	return weather_df

def fetch_isw_report_html(date_str, debug = False):
	BASE_URL = "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment"
	months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
	date_arr = date_str.split("-")

	url = f"{BASE_URL}-{months[int(date_arr[1])-1]}-{int(date_arr[2])}-{int(date_arr[0])}"
	if debug:
		print(url)

	page = requests.get(url)
	html = BeautifulSoup(page.content, features="html.parser")
	title = html.title.string

	if "404" in title:
		return ""
	else:
		return page.content


def get_isw_report_html(date_str, find_latest_if_not_available = False, debug = False):
	isw_report_html_dir = f"{OUTPUT_FOLDER}/isw"
	isw_report_html_file_name = f"isw___{date_str}.html"

	if os.path.isfile(f"{isw_report_html_dir}/{isw_report_html_file_name}"):
		if debug:
			print(f"ISW report for {date_str} is downloaded. Reading...")

		with open(f"{isw_report_html_dir}/{isw_report_html_file_name}", "r", encoding="utf8") as cfile:
			html = BeautifulSoup(cfile.read(), features="html.parser")
			content_html = str(html.body.find("div", attrs={"class": "field-type-text-with-summary"}))
			content_html = isw_preprocessing.preprocess_page_html(content_html)
			return content_html
	else:
		content = fetch_isw_report_html(date_str, debug)

		if content == "":
			if find_latest_if_not_available:
				original_date_str = date_str
				date_str = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(1)).strftime("%Y-%m-%d")
				if debug:
					print(f"ISW report for {original_date_str} is not available. Trying {date_str}")
				return get_isw_report_html(date_str, find_latest_if_not_available, debug)
			else:
				raise Exception(f"No ISW report found for {date_str}")
		else:
			if debug:
				print(f"ISW report for {date_str} is available. Saving...")

			if not os.path.exists(isw_report_html_dir):
				os.makedirs(isw_report_html_dir)

			isw_report_html_file_name = f"isw___{date_str}.html"
			with open(f"{isw_report_html_dir}/{isw_report_html_file_name}", "wb+") as f:
				f.write(content)
			
			html = BeautifulSoup(content, features="html.parser")
			content_html = str(html.body.find("div", attrs={"class": "field-type-text-with-summary"}))
			content_html = isw_preprocessing.preprocess_page_html(content_html)
			return content_html


def get_prediction_for_date(region, date_str, model, debug = False):
	weather_df = get_weather(region, date_str, debug)
	weather_df_v2 = pd.merge(weather_df, df_regions, left_on="city", right_on="center_city_ua")

	isw_report_html = get_isw_report_html(date_str, True, debug)
	content_text_lemm = text_preprocessing.text_preprocess(isw_report_html, "lemm")

	word_count_vector = cv.transform([content_text_lemm])
	tfidf_vector = tfidf.transform(word_count_vector)

	df_work_v2 = weather_df_v2.drop(weather_exclude, axis=1, errors='ignore')
	df_work_v2 = df_work_v2.drop(fields_to_exclude, axis=1, errors='ignore')
	df_work_v2["hour_conditions"] = df_work_v2["hour_conditions"].apply(lambda x: x.split(",")[0])
	df_work_v2["hour_conditions_id"] = label_encoder.transform(df_work_v2["hour_conditions"])
	df_work_v3 = df_work_v2.drop(tmp_fields_to_exclude, axis=1, errors='ignore')

	tfidf_matrix = tfidf_vector
	for i in range(0, 23):
		tfidf_matrix = sp.vstack((tfidf_matrix, tfidf_vector), format="csr")
	
	df_work_v4_csr = sp.csr_matrix(df_work_v3.values)
	df_all_features = sp.hstack((df_work_v4_csr, tfidf_matrix), format="csr")
	
	hours_alarm_schedule = model.predict(df_all_features)
	return hours_alarm_schedule

def get_prediction_for_next_12_hours(region, model, debug = False):
	current_dateTime = datetime.now()
	ts_start = current_dateTime.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
	start_date = ts_start.strftime("%Y-%m-%d")

	ts_end = ts_start + timedelta(hours=12)
	end_date = ts_end.strftime("%Y-%m-%d")

	schedule = []

	if (start_date == end_date):
		if debug:
			print(f"Preparing alarms forecast for {start_date}: {ts_start.hour}:00 - {ts_end.hour - 1}:00")

		today_forecast = get_prediction_for_date(region, start_date, model, debug)
		time_forecast = today_forecast[ts_start.hour : ts_end.hour]

		for idx, hour in enumerate(range(ts_start.hour, ts_end.hour)):
			schedule.append({ f"{hour}:00": "false" if time_forecast[idx] == 0 else "true" })
	else:
		if debug:
			print(f"Preparing alarms forecast for two days: {ts_start} - {ts_end - timedelta(hours=1)}")

		today_forecast = get_prediction_for_date(region, start_date, model, debug)
		tomorrow_forecast = get_prediction_for_date(region, end_date, model, debug)

		today_time_forecast = today_forecast[ts_start.hour:]
		tomorrow_time_forecast = tomorrow_forecast[:ts_end.hour]

		for idx, hour in enumerate(range(ts_start.hour, 24)):
			schedule.append({ f"{hour}:00": "false" if today_time_forecast[idx] == 0 else "true" })
		for idx, hour in enumerate(range(0, ts_end.hour)):
			schedule.append({ f"{hour}:00": "false" if tomorrow_time_forecast[idx] == 0 else "true" })

	return schedule

# print(get_prediction_for_date("Крим", "2023-04-24", MLP, True))
print(get_prediction_for_next_12_hours("Крим", MLP, True))
