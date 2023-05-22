import pytest
from pyssl.methods.SmSCluster import SmSCluster
from pyssl.ssl_stream import SemiSupervisedStream
from river import metrics as rv_metrics


@pytest.mark.usefixtures("weather_stream")
def test_SmSCluster(weather_stream):
    data_size = 0
    unlabeled = 0
    labeled = 0
    proportion = 0.05
    accuracy = rv_metrics.Accuracy()
    stream = SemiSupervisedStream(weather_stream, 0.05, 42, 200)

    tree = SmSCluster()

    for _ in range(200):
        data, label, _ = next(stream)
        tree.learn_one(data, label)

    for x, y, true_y in stream:
        if y is None:
            unlabeled += 1
        else:
            labeled += 1

        predict = tree.predict_one(x)
        accuracy.update(true_y, predict)
        tree.learn_one(x, y)

        data_size = data_size + 1

    assert labeled / (labeled + unlabeled) == pytest.approx(proportion, 0.1)
    assert accuracy.get() == pytest.approx(0.68, 0.1)
