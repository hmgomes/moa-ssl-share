# from skmultiflow.bayes import NaiveBayes
import pytest
from conftest import WEATHER_CSV
from pyssl.methods.CPSSDS import CPSSDS

from river import metrics as rv_metrics
from river.stream import iter_csv
import numpy as np

# import numpy as np

from pyssl.ssl_stream import SemiSupervisedStream

CPSSDS_EXPECTATIONS = [
    (200, 0.1, 0.98, "NaiveBayes", 0.38),
    (500, 0.05, 0.98, "NaiveBayes", 0.33),
    (1000, 0.05, 0.98, "NaiveBayes", 0.62),
    (1000, 0.05, 0.98, "HoeffdingTree", 0.64),
]


def test_CPSSDS_nb():
    """Test CPSSDS with NaiveBayes"""

    # Set seed
    np.random.seed(42)

    for (
        chunk_size,
        label_budget,
        significance_level,
        algorithm,
        expectation,
    ) in CPSSDS_EXPECTATIONS:
        weather_stream = iter_csv(
            WEATHER_CSV, target="target", converters={"target": int}
        )
        weather_ssl_stream = SemiSupervisedStream(weather_stream, label_budget, 42)
        accuracy = rv_metrics.Accuracy()
        ssl_method = CPSSDS(algorithm, chunk_size, 2, significance_level)

        for x, y, y_true in weather_ssl_stream:
            y_hat = ssl_method.predict_one(x)
            accuracy.update(y_true, y_hat)
            ssl_method.learn_one(x, y)

        assert accuracy.get() == pytest.approx(expectation, 0.1)
