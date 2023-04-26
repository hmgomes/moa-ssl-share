"""
Tests ensure consistency between python and java. These tests have a java
counterpart moa/src/main/java/moa/streams/SemiSupervisedStream.java
"""

import typing as t

import numpy as np
import pytest
from river import stream as rs
from sklearn import datasets

from pyssl.ssl_stream import SemiSupervisedStream
from river.base.typing import Dataset

from pyssl.skmf import convert_river_to_skmf

def test_RNG_consistency():
    """Ensure the PRNG is seeded correctly, so that the stream
    is deterministic and consistent with other languages.
    """
    expected = [0.37454, 0.95071, 0.73199, 0.59865, 0.15601,
                0.15599, 0.05808, 0.86617, 0.60111, 0.70807]
    mt19937 = np.random.MT19937()
    mt19937._legacy_seeding(42)
    rand = np.random.Generator(mt19937)
    for i in range(10):
        # Assert equals with delta
        assert rand.random(dtype=np.float64) == pytest.approx(
            expected[i], 0.0001)


@pytest.mark.parametrize("warmup_length", [0, 100])
@pytest.mark.parametrize("delay", [None, 0, 1, 5, 100])
def test_ssl_stream(warmup_length, delay):
    """Ensure EvaluateInterleavedTestThenTrainSSLDelayed returns a 
    stream consistent with other implementations, of semi-supervised
    learning, when the PRNG is seeded. This is to ensure that the 
    stream is deterministic and consistent.
    """

    expected_y = {
        None: [0, None, None, None, 4, 5, 6, None, None, None],
        0: [0, None, 1, None, 2, None, 3, 4, 5, 6, None, 7, None, 8, None],
        1: [0, None, None, 1, 2, None, 4, 3, 5, 6, None, None, 7, 8, None],
        5: [0, None, None, None, 4, 5, 6, 1, 2, 3, None, None, None],
        100: [0, None, None, None, 4, 5, 6, None, None, None]
    }[delay]

    stream = zip(range(10 + warmup_length), range(10 + warmup_length))
    sss = SemiSupervisedStream(stream, 0.5, 42, warmup_length, delay)

    for i, (_, y, true_y) in enumerate(sss):
        if i < warmup_length:
            assert y == i, "Instances during warmup should always have a label"
            continue
        
        # Ensure the label is correct and in the correct order
        if expected_y[i - warmup_length] is not None:
            assert y - warmup_length == expected_y[i - warmup_length]
        else:
            assert y is None

    assert i + 1 == len(expected_y) + warmup_length



def test_probability():
    """Ensure the frequency of labeled vs unlabeled instances
    is approximately correct.
    """
    stream = zip(range(10000), range(10000))
    sss = SemiSupervisedStream(stream, 0.8, 42)

    num_labeled = 0
    num_unlabeled = 0
    for i, (x, y, true_y) in enumerate(sss):
        has_label = y is not None
        if has_label:
            num_labeled += 1
        else:
            num_unlabeled += 1

    assert num_labeled == 8038
    assert num_unlabeled == 1962

@pytest.fixture
def ssl_real_stream():
    stream = rs.iter_sklearn_dataset(datasets.load_breast_cancer())
    sss = SemiSupervisedStream(stream, 0.8, 42, 0)
    return sss



def test_river_compatibility():
    """Ensure SemiSupervisedStream is compatible with River.
    """
    stream = rs.iter_sklearn_dataset(datasets.load_breast_cancer())
    sss = SemiSupervisedStream(stream, 0.8, 42, 0)

    unlabeled = 0
    labeled = 0
    for xi, yi, true_y in sss:
        if yi is None:
            unlabeled += 1
        else:
            labeled += 1

    assert labeled == 442
    assert unlabeled == 127

def test_exhaustible():
    """Ensure SemiSupervisedStream is exhaustible.
    """
    stream = rs.iter_sklearn_dataset(datasets.load_breast_cancer())
    sss = SemiSupervisedStream(stream, 0.8, 42, 0)

    for xi, yi, true_y in sss:
        pass

    with pytest.raises(StopIteration):
        next(sss)
    
    for xi, yi, true_y in sss:
        raise Exception("Should not be reached")


def test_streaming_take(ssl_real_stream: Dataset):
    for x, y, _ in ssl_real_stream:
        x, y = convert_river_to_skmf(x, y)
        assert x.shape == (1, 30)

