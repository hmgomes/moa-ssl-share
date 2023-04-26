"""
Added compatibility with scikit-multiflow.
"""

import typing as t

import numpy as np

from pyssl.ssl_stream import SemiSupervisedLabel


def convert_river_to_skmf(x: dict, y: SemiSupervisedLabel = None) -> t.Tuple[np.ndarray, np.ndarray]:
    """Convert a single instance into a numpy array. For use with scikit-multiflow.

    :param x: The instance to convert.
    :param y: The label of the instance.
    :return: A tuple of numpy arrays, the first containing the instances and
        the second containing the labels.
    """
    numpy_x = np.array([list(map(float, x.values()))])

    if y is None:
        numpy_y = np.array([int(-1)])
    else:
        numpy_y = np.array([int(y)])
    return numpy_x, numpy_y
