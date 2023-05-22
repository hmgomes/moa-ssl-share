import os
import urllib.request

import pytest

from pyssl.ssl_stream import SemiSupervisedStream
from river.stream import iter_csv


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


@pytest.fixture
def weather_stream():
    stream = iter_csv(WEATHER_CSV, target="target", converters={"target": int})
    return stream


@pytest.fixture
def weather_ssl_stream():
    stream = iter_csv(WEATHER_CSV, target="target", converters={"target": int})
    return SemiSupervisedStream(stream, 0.05, 42)
