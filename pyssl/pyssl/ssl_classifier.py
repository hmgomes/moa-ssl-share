import abc

from river.base import Classifier

from pyssl.ssl_stream import SemiSupervisedLabel


class SemiSupervisedClassifier(Classifier):
    @abc.abstractmethod
    def learn_one(self, x: dict, y: SemiSupervisedLabel) -> "SemiSupervisedClassifier":
        """Update the model with a set of features `x` and no label.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        self

        """
        raise NotImplementedError
