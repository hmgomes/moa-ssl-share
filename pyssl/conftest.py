import os
import urllib.request

def setup_weather_dataset():
    # Load weather.csv from
    # https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/weather.csv
    path = "test_datasets/weather.csv"

    # If the file exists, return the path
    if os.path.exists(path):
        return path

    print(f"Weather dataset not found at {path}, downloading...")
    # Otherwise, download the file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/weather.csv"
    urllib.request.urlretrieve(url, path)

    return path

WEATHER_CSV = setup_weather_dataset()
