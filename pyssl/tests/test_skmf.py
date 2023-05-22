import numpy as np
import river.stream as rs
from pyssl.skmf import (
    ChunkQueue,
    ClassifierBatched,
    LengthMismatchError,
    convert_river_to_skmf,
    convert_skmf_to_river,
    shuffle_split,
    split_by_label_presence,
)

# from skmultiflow.data.file_stream import FileStream
from river.naive_bayes import GaussianNB
from conftest import WEATHER_CSV
import pytest


# def test_conversion():
#     skmf_stream = FileStream(WEATHER_CSV)
#     river_stream = rs.iter_csv(WEATHER_CSV, target="target")

#     i = 0
#     while skmf_stream.has_more_samples() and i < 1_000:
#         skmf_x, skmf_y = skmf_stream.next_sample()
#         river_x, river_y = next(river_stream)
#         x, y = convert_river_to_skmf(river_x, river_y)
#         i += 1

#         assert np.allclose(x, skmf_x)
#         assert y == skmf_y


def test_ChunkQueue():
    chunk_size = 2
    stream_chunk = ChunkQueue(chunk_size)

    instance1 = {"feat1": 1.0, "feat2": 2.0}
    instance2 = {"feat1": 3.0, "feat2": 4.0}
    instance3 = {"feat1": 5.0, "feat2": 6.0}
    instance4 = {"feat1": 7.0, "feat2": 8.0}

    # Add instances to the stream chunk and check if the chunk is ready
    stream_chunk.add(instance1, 1)
    assert not stream_chunk.chunk_ready()
    stream_chunk.add(instance2, 2)
    assert stream_chunk.chunk_ready()
    stream_chunk.add(instance3, None)
    assert stream_chunk.chunk_ready()

    # Take the chunk and check if we receive the first instances
    chunk_x, chunk_y = stream_chunk.take_chunk()
    assert np.array_equal(chunk_x, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(chunk_y, np.array([1, 2]))

    # When the chunk is empty, we should not be able to take a chunk
    assert not stream_chunk.chunk_ready()
    with pytest.raises(ValueError):
        stream_chunk.take_chunk()

    # Add more instances and check if the chunk is ready
    stream_chunk.add(instance4, 4)
    assert stream_chunk.chunk_ready(), f"Chunk size is {len(stream_chunk)}"
    chunk_x, chunk_y = stream_chunk.take_chunk()
    assert np.array_equal(chunk_x, np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert np.array_equal(chunk_y, np.array([-1, 4]))


def test_split_by_label_presence():
    # Create a test input
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, -1, 0])

    # Expected output
    expected_labeled_instances = np.array([[1, 2, 3], [7, 8, 9]])
    expected_labeled_labels = np.array([1, 0])
    expected_unlabeled_instances = np.array([[4, 5, 6]])

    # Call the function being tested
    (x_l, y_l), x_unlabeled = split_by_label_presence(x, y)

    # Check the output types and shapes
    assert isinstance(x_l, np.ndarray)
    assert isinstance(y_l, np.ndarray)
    assert isinstance(x_unlabeled, np.ndarray)

    # Check the output values
    np.testing.assert_array_equal(x_l, expected_labeled_instances)
    np.testing.assert_array_equal(y_l, expected_labeled_labels)
    np.testing.assert_array_equal(x_unlabeled, expected_unlabeled_instances)


def test_LengthMismatchError():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2, 3])

    with pytest.raises(LengthMismatchError):
        split_by_label_presence(x, y)

    with pytest.raises(LengthMismatchError):
        shuffle_split(0.2, x, y)


def test_shuffle_split():
    # Seed numpy
    np.random.seed(42)
    x = np.arange(100)
    y = np.arange(100)
    proportion = 0.2

    (x_a, y_a), (x_b, y_b) = shuffle_split(proportion, x, y)

    assert len(x_a) == len(y_a)
    assert len(x_b) == len(y_b)

    assert len(x_a) + len(x_b) == len(x)
    assert len(y_a) + len(y_b) == len(y)

    # Check proportions
    assert len(x_a) / len(x) == proportion
    assert len(x_b) / len(x) == 1 - proportion


def test_convert_skmf_to_river():
    # Sample input data
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    y = np.array([0, 1, -1])

    # Expected output
    expected_output = [
        ({0: 1, 1: 2, 2: 3}, 0),
        ({0: 4, 1: 5, 2: 6}, 1),
        ({0: 7, 1: 8, 2: 9}, None),
    ]

    river_sequence = convert_skmf_to_river(x, y)

    # Check the output values
    assert river_sequence == expected_output

    for i in range(len(river_sequence)):
        x_out, y_out = convert_river_to_skmf(*river_sequence[i])
        assert np.array_equal(x_out, [x[i]])
        assert y_out == y[i]


def test_ClassifierBatchAdaptor():
    classifier = ClassifierBatched(GaussianNB())

    # Random data
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    print(y.shape)
    classifier.learn_many(x, y)

    assert classifier.predict_proba_many(x).shape == (100, 2)
