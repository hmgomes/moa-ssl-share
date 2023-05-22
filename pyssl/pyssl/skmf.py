"""
Added compatibility with scikit-multiflow.
"""

import typing as t

import numpy as np
from river.base.classifier import Classifier

from pyssl.ssl_stream import SemiSupervisedLabel
from river.base.typing import ClfTarget


class LengthMismatchError(ValueError):
    pass


def convert_river_to_skmf(
    x: dict, y: SemiSupervisedLabel = None
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Convert a single instance into a numpy array. For use with scikit-multiflow.

    :param x: The instance to convert.
    :param y: The label of the instance. If None, the instance is unlabeled.
    :return: A tuple of numpy arrays, the first containing the instances and
        the second containing the labels.
    """

    if y is None:
        numpy_y = np.array([int(-1)])
    else:
        numpy_y = np.array([int(y)])
    numpy_x = np.array([list(map(float, x.values()))])
    return numpy_x, numpy_y


def convert_one_skmf_to_river(
    x: np.ndarray, y: t.Optional[np.ndarray] = None
) -> t.Tuple[dict, SemiSupervisedLabel]:
    """Convert a single scikit-multiflow instance into a river instance.

    :param x: A single scikit-multiflow instance.
    :param y: A single scikit-multiflow label.
    :return: A river instance. The keys of the dictionary are the indices of
        the features.
    """
    x_elem_dict = {}
    for i, x_elem_val in enumerate(x):
        x_elem_dict[i] = x_elem_val

    # -1 denotes an unlabeled instance
    if y == -1:
        y = None

    return x_elem_dict, y


def convert_skmf_to_river(
    x: np.ndarray, y: t.Optional[np.ndarray] = None
) -> t.Sequence[t.Tuple[dict, SemiSupervisedLabel]]:
    """Convert a batch of scikit-multiflow instances into a list of river instances.

    :param x: A batch of scikit-multiflow instances.
    :param y: A batch of scikit-multiflow labels.
    :raises LengthMismatchError: The length of x and y must be the same.
    :return: A list of river instances. The keys of the dictionary are the
        indices of the features.
    """
    if y is None:
        y = [-1] * len(x)
    if len(x) != len(y):
        raise LengthMismatchError("x and y must have the same length")

    result = []
    for x_i, y_i in zip(x, y):
        result.append(convert_one_skmf_to_river(x_i, y_i))
    return result


def split_by_label_presence(
    x: np.ndarray, y: np.ndarray
) -> t.Tuple[t.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Split the data into labeled and unlabeled instances.

    :param x: A batch of instances.
    :param y: A batch of labels where -1 means that the instance is unlabeled.
    :raises LengthMismatchError: The length of x and y must be the same.
    :return:
        - A tuple containing the labeled instances and labels.
        - A numpy array containing the unlabeled instances.
    """
    if len(x) != len(y):
        raise LengthMismatchError("x and y must have the same length")
    labeled_mask = y != -1
    return (x[labeled_mask], y[labeled_mask]), x[~labeled_mask]


def shuffle_split(
    split_proportion: float, x: np.ndarray, y: np.ndarray
) -> t.Tuple[t.Tuple[np.ndarray, np.ndarray], t.Tuple[np.ndarray, np.ndarray]]:
    """Shuffle and split the data into two parts.

    :param split_proportion: The proportion of the dataset to be included in
        the first part.
    :param x: The instances to split.
    :param y: The labels to split.
    :raises LengthMismatchError: The length of x and y must be the same.
    :return: Two tuples containing the instances and labels of the two parts.
    """
    if len(x) != len(y):
        raise LengthMismatchError("x and y must have the same length")
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    split_index = int(len(x) * split_proportion)
    idx_a = indices[:split_index]
    idx_b = indices[split_index:]
    return (x[idx_a], y[idx_a]), (x[idx_b], y[idx_b])


class ChunkQueue:
    """This class is used to adapt an incremental river stream into a series of
    scikit-multiflow chunks.
    """

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.buffer_x: t.Optional[np.ndarray] = None
        self.buffer_y: t.Optional[np.ndarray] = None

    def add(self, x: dict, y: SemiSupervisedLabel = None) -> None:
        """Add a new instance to the chunk.

        :param x: The instance to add.
        :param y: The label of the instance.
        """
        numpy_x, numpy_y = convert_river_to_skmf(x, y)

        # Initialize the chunk if it is empty.
        if self.buffer_x is None or self.buffer_y is None:
            self.buffer_x = numpy_x
            self.buffer_y = numpy_y
            return

        self.buffer_x = np.append(self.buffer_x, numpy_x, axis=0)
        self.buffer_y = np.append(self.buffer_y, numpy_y, axis=0)

    def chunk_ready(self) -> bool:
        """Check if the chunk is full.

        :return: True if the chunk is full, False otherwise.
        """
        return len(self) >= self.chunk_size

    def __len__(self) -> int:
        assert len(self.buffer_x) == len(
            self.buffer_y
        ), "Buffers are not the same size."
        return len(self.buffer_x)

    def take_chunk(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Take the chunk and reset the chunk.

        :return: A tuple containing the instances and labels in the chunk.
        """
        if not self.chunk_ready():
            raise ValueError("Chunk is not ready.")

        chunk_x = self.buffer_x[: self.chunk_size]
        chunk_y = self.buffer_y[: self.chunk_size]
        self.buffer_x = self.buffer_x[self.chunk_size :]
        self.buffer_y = self.buffer_y[self.chunk_size :]
        return chunk_x, chunk_y


class ClassifierBatched(Classifier):
    def __init__(self, classifer: Classifier) -> None:
        super().__init__()
        if not isinstance(classifer, Classifier):
            raise TypeError("The classifer must be a subclass of river.base.Classifier")
        self.classifier = classifer

    def learn_one(self, x: dict, y: ClfTarget) -> Classifier:
        return self.classifier.learn_one(x, y)

    def predict_one(self, x: dict) -> ClfTarget:
        return self.classifier.predict_one(x)

    def predict_proba_one(self, x: dict) -> dict:
        return self.classifier.predict_proba_one(x)

    def learn_many(self, x: np.ndarray, y: np.ndarray):
        if len(x) != len(y):
            raise LengthMismatchError("x and y must have the same length")
        for x_i, y_i in convert_skmf_to_river(x, y):
            self.learn_one(x_i, y_i)

    def predict_proba_many(self, x: np.ndarray) -> np.ndarray:
        results = []
        for x_i, _ in convert_skmf_to_river(x):
            y_hat = self.predict_proba_one(x_i)
            y_hat_skmf = np.array(list(y_hat.values()))
            # print(y_hat_skmf)
            results.append(y_hat_skmf)
        return np.stack(results)
