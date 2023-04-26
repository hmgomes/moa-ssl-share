import pytest
from pyssl.methods.SmSCluster import SmSCluster
from pyssl.ssl_stream import SemiSupervisedStream
from river import stream as rs
from conftest import WEATHER_CSV

@pytest.fixture
def weather_stream():
    stream = rs.iter_csv(
        WEATHER_CSV, 
        target="target",
        converters={"target": int}
    )
    return stream

def test_SmSCluster(weather_stream):
    count = 0
    data_size = 0
    unlabeled = 0
    labeled = 0
    stream = SemiSupervisedStream(weather_stream, 0.1, 42, 200)

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
        if predict == true_y:
            count = count + 1

        tree.learn_one(x, y)

        data_size = data_size + 1


    assert count/data_size >= 0.5
    assert labeled/(labeled+unlabeled) == pytest.approx(0.1, 0.1)
        

