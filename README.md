<p align="center" style="font-size: 40px">Air Alarms</p>
<p align="center" style="font-size: 20px">A system for shellings prediction in Ukraine</p>

<br/>

<p align="center">
	This project is crucial as it leverages data from news sources, such as ISW reports, weather information, and the ongoing situation in neighboring regions, to provide early warnings and enhance our understanding of potential conflict escalation. By developing this tool, we aim to support decision-makers in crafting informed strategies, ultimately promoting global stability and peace.
</p>

<br/>

## Running prediction

First, download and install [Python 3.11](https://www.python.org/downloads/).

Create a `.env` file with the [VisualCrossing Weather](https://www.visualcrossing.com/weather-api) API key in the following format:
```
WEATHER_API_KEY=<your key>
```

Setup all the deps listed in `requirements.txt`:
```
pip3 install -r requirements.txt
```

Run the prediction script for all regions for the next 12 hours:
```
python3 prediction.py
```

Observe the output in `prediction.json`.

<br/>

## Running a server

The system can also be run as a server. To do so, first install uWSGI:
```
pip3 install uwsgi
```
and run the server:
```
uwsgi --http 0.0.0.0:8000 --wsgi-file server.py --callable app --processes 4 --threads 2
```
The server has two endpoints. The first one is for manually triggering the prediction script:
```
POST /api/prediction/v1/update_forecast
```
Another endpoint is for grabbing the generated data:
```
GET /api/prediction/v1/predict
```
with the desired location in the request body:
```
{
    "locations": "Kyiv"
}
```
or all locations at once:
```
{
    "locations": "*"
}
```
Additionally, you can configure a CRON task to run the prediction script every hour by running `cron.sh`. 

<br/>

## Components

### Jupyter Notebook files

Files ending with `.ipynb` were created as a demonstration of the full process of collecting, processing, and applying the data from different sources: ISW reports, weather data, historical alarms data etc. In fact, these files are also used to prepare everything for training models.

### Python script files

Files ending with `.py` are executables used in production.
- `3_models.py` handles training different models
- `prediction.py` is responsible for generating future air alarms predictions making use of created models and the latest ISW report + weather forecast data
- `server.py` holds the server code that can update the prediction data and reveal it to public
- Utility files under the `utils` folder are used in both notebook files and top level scripts

<br/>

## Credits

Pavlo Kolinko

Vladyslav Tkalenko

Vadym Ohyr

Oleksandr Parkhomchuk

Teacher: Andew Kurochkin