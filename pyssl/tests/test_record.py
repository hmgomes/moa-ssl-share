from pyssl.methods.record.record import Record
import h5py
import numpy as np
from river import stream as rv_stream
from river import metrics as rv_metrics
import pytest
from tqdm import tqdm
from pyssl.ssl_stream import SemiSupervisedStream

data_filename = "/home/antonlee/github.com/tachyonicClock/moa-ssl-share/pyssl/test_datasets/2CDT.mat"


@pytest.fixture
def two_cdt_stream():
    data = h5py.File(data_filename, mode="r")
    x_data = np.transpose(data["x"])
    n_features: int = x_data.shape[1]

    y_labels = np.transpose(data["y"]).astype(int)
    y_labels[y_labels == -1] = 2  # Convert labels from {-1, 1} to {1, 2}
    class_labels = [1, 2]

    # Create a semi-supervised stream
    ssl_stream = SemiSupervisedStream(
        rv_stream.iter_array(x_data, y_labels.ravel()), 0.01, 42
    )
    return (ssl_stream, n_features, class_labels)


@pytest.mark.parametrize("method,expected", [("S3VM", 0.8), ("MT", 0.88), ("LP", 0.9)])
def test_record(two_cdt_stream, method: str, expected: float):
    ssl_stream, n_features, class_labels = two_cdt_stream
    record = Record(280, 100, n_features, method, class_labels)

    accuracy = rv_metrics.Accuracy()
    for x, y, true_y in tqdm(ssl_stream):
        if record.is_classifier_ready():
            accuracy.update(true_y, record.predict_one(x))
        record.learn_one(x, y)

    assert accuracy.get() == pytest.approx(expected, 0.01)


@pytest.mark.usefixtures("weather_stream")
@pytest.mark.parametrize(
    "method,expected", [("S3VM", 0.68), ("MT", 0.63), ("LP", 0.68)]
)
def test_record_weather(weather_stream, method, expected):
    limit = 1000
    stream = SemiSupervisedStream(weather_stream, 0.05, 42)
    accuracy = rv_metrics.Accuracy()

    record = Record(100, 200, 8, method, [0, 1])
    for i, (x, y, true_y) in enumerate(tqdm(stream)):
        if record.is_classifier_ready():
            accuracy.update(true_y, record.predict_one(x))
        record.learn_one(x, y)
        if i > limit:
            break

    assert accuracy.get() == pytest.approx(
        expected, 0.05
    ), f"RECORD({method}) failed to achieve {expected} accuracy on weather dataset."
