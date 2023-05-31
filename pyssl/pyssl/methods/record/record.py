"""
Derived from on https://github.com/WNJXYK/RECORD and
Record: Resource constrained semi-supervised learning under distribution shift
Guo, Lan-Zhe, Zhi Zhou, and Yu-Feng Li. "Record: Resource constrained
semi-supervised learning under distribution shift." Proceedings of the
26th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining. 2020.
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import typing as t

from sklearn.semi_supervised import LabelPropagation
from pyssl.methods.record.DSSL import MT
from pyssl.methods.record.S3VM import S3VM
from pyssl.skmf import (
    ChunkQueue,
    convert_river_to_skmf,
)

from pyssl.ssl_classifier import SemiSupervisedClassifier
from pyssl.ssl_stream import SemiSupervisedLabel


def push(
    lX: np.ndarray, ly: np.ndarray, budget: int, X: np.ndarray, y: float
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Push adds a single new data point to the buffer of data features and
    labels. If the buffer is full, the oldest data point is removed.

    :param lX: The buffer of data features to be updated
    :param ly: The buffer of data labels to be updated
    :param budget: The maximum size of the buffer per class
    :param X: New data features
    :param y: The label of the new data features
    :return: An updated buffer of data features and labels of length
        <=(budget * class count)
    """
    ix = ly[:, 0] == y
    cur_X, cur_y = lX[ix], ly[ix]
    lX, ly = lX[~ix], ly[~ix]
    while cur_X.shape[0] >= budget:
        cur_X = cur_X[1:, :]
        cur_y = cur_y[1:, :]
    cur_X = np.vstack([cur_X, X.reshape(1, -1)])
    cur_y = np.vstack([cur_y, y.reshape(1, -1)])
    lX, ly = np.vstack([lX, cur_X]), np.vstack([ly, cur_y])
    return lX, ly


def generate_labels(n, c):
    return np.array([c] * n).reshape(-1, 1)


def random_item(X, m):
    if X.shape[0] < m:
        return X[np.random.choice(np.arange(X.shape[0]), m, replace=True)]
    return X[np.random.choice(np.arange(X.shape[0]), m, replace=False)]


class IFBinaryLogistic:
    def __init__(self):
        self.model = None
        self.scaler = None

    def sigmod(self, x):
        return 1.0 / (1 + np.exp(x))

    def train(self, X, y):
        C = 0.1 / X.shape[0]
        self.model = LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False,
            solver="lbfgs",
            warm_start=True,
            max_iter=10000,
        )
        self.model.fit(X, y)
        self.theta = self.model.coef_

    def get_influence(self, train_X, test_X, train_y, test_y):
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(np.vstack([train_X, test_X]))
        train_X = self.scaler.transform(train_X)
        test_X = self.scaler.transform(test_X)

        self.train(train_X, train_y)

        n, p = train_X.shape[0], train_X.shape[1]

        hessian = np.zeros((p, p))
        for i in range(int(n)):
            X, y = train_X[i].reshape((-1, 1)), train_y[i]
            hessian += (
                self.sigmod(np.dot(self.theta, X))
                * self.sigmod(-np.dot(self.theta, X))
                * np.dot(X, X.transpose())
            )
        hessian /= n
        ihessian = pinv(hessian)

        influence = np.zeros(n)
        test_X, test_y = test_X.transpose(), test_y
        for i in range(n):
            X, y = train_X[i].reshape((-1, 1)), train_y[i]
            partial_X = -self.sigmod(-y * np.dot(self.theta, X)) * y * X
            partial_test = -np.dot(
                np.multiply(
                    test_y, self.sigmod(np.multiply(test_y, np.dot(self.theta, test_X)))
                ),
                test_X.transpose(),
            )
            influence[i] = -np.dot(np.dot(partial_test, ihessian), partial_X)

        return influence

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return accuracy_score(self.predict(X), y)


def RECORDS_choice(
    lX: np.ndarray,
    ly: np.ndarray,
    X: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    buffer_budget: int,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Use RECORD to update the buffer of data features and labels with
    the unlabeled data features and psuedo-labels.

    :param lX: The buffer of data features to be updated
    :param ly: The buffer of data labels to be updated
    :param X: The unlabeled data features to be selected from
    :param pred: The psuedo-labels of the unlabeled data features
    :param proba: The probability of the psuedo-labels
    :param buffer_budget: Total memory budget
    :return: An updated buffer of data features and labels of length
    """
    proba = proba.max(axis=1)
    new_X, new_pred, new_proba = X, pred, proba

    classes = list(set(list(new_pred)))
    class_budget = int(buffer_budget / len(classes))

    prev_X, prev_y = np.zeros((0, X.shape[1])), np.zeros((0, 1))
    curr_X, curr_y = np.zeros((0, X.shape[1])), np.zeros((0, 1))
    curr_p = np.zeros((0, 1))
    for c in classes:
        ix = [new_pred[i] == c and new_proba[i] > 0.6 for i in range(new_X.shape[0])]
        n = int(np.sum(ix))
        sorted_X = new_X[ix][np.argsort(-new_proba[ix])]
        sorted_proba = new_proba[ix][np.argsort(-new_proba[ix])]
        div_point, _ = int(n / 2), 0
        prev_X, curr_X = np.vstack([prev_X, sorted_X[div_point:]]), np.vstack(
            [curr_X, sorted_X[:div_point]]
        )
        prev_y, curr_y = np.vstack(
            [prev_y, generate_labels(n - div_point, c)]
        ), np.vstack([curr_y, generate_labels(div_point, c)])
        curr_p = np.vstack([curr_p, sorted_proba[:div_point].reshape(-1, 1)])
    prev_y, curr_y = prev_y.ravel(), curr_y.ravel()

    for c in classes:
        n_tra, n_tst = np.sum(curr_y == c), np.sum(prev_y == c)
        if (
            n_tra == 0
            or n_tst == 0
            or np.sum(curr_y != c) == 0
            or np.sum(prev_y != c) == 0
        ):
            sorted_X = np.vstack([curr_X[curr_y == c], prev_X[prev_y == c]])
            n = min(class_budget, n_tst + n_tra)
        else:
            tra_X = np.vstack(
                [curr_X[curr_y == c], random_item(curr_X[curr_y != c], n_tra)]
            )
            tst_X = np.vstack(
                [prev_X[prev_y == c], random_item(prev_X[prev_y != c], n_tst)]
            )
            tra_y = np.vstack(
                [generate_labels(n_tra, 1), generate_labels(n_tra, -1)]
            ).ravel()
            tst_y = np.vstack(
                [generate_labels(n_tst, 1), generate_labels(n_tst, -1)]
            ).ravel()

            clf = IFBinaryLogistic()
            influence = clf.get_influence(tra_X, tst_X, tra_y, tst_y)[:n_tra]
            # print(influence)
            sorted_X = curr_X[curr_y == c][np.argsort(influence)]
            sorted_proba = curr_p[curr_y == c][np.argsort(influence)]
            n = min(class_budget, min(n_tra, np.sum(influence < 0)))

        for i in range(n - 1, -1, -1):
            lX, ly = push(lX, ly, class_budget, sorted_X[i], c)

    return lX, ly


class RecordBuffer:
    def __init__(self, budget: int, features: int) -> None:
        self.x_buffer = np.zeros((0, features))
        self.y_buffer = np.zeros((0, 1))
        self.features = features
        self.budget = budget

    def update(
        self,
        unlabeled_x: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        if (
            unlabeled_x.shape[0] != predictions.shape[0]
            or unlabeled_x.shape[0] != probabilities.shape[0]
        ):
            raise ValueError(
                "`unlabeled_x`, `predictions` and `probabilities` must have "
                + "the same number of elements got {0}, {1}, {2} respectively".format(
                    unlabeled_x.shape[0], predictions.shape[0], probabilities.shape[0]
                )
            )
        if unlabeled_x.shape[1] != self.features:
            raise ValueError(
                "`unlabeled_x` must have {0} features got {1}".format(
                    self.features, unlabeled_x.shape[1]
                )
            )

        self.x_buffer, self.y_buffer = RECORDS_choice(
            self.x_buffer,
            self.y_buffer,
            unlabeled_x,
            predictions,
            probabilities,
            self.budget,
        )

    def add_labeled_instance(self, x: np.ndarray, y: float) -> None:
        if x.shape[1] != self.features:
            raise ValueError(
                "`x` must have {0} features got {1}".format(self.features, x.shape[0])
            )
        self.x_buffer = np.vstack([self.x_buffer, x])
        self.y_buffer = np.vstack([self.y_buffer, y])


class Record(SemiSupervisedClassifier):
    """RECORD: Resource Constrained Semi-Supervised Learning under Distribution
    Shift [0]

    Record maintains a buffer of true labeled and pseudo-labeled data points.
    The buffer is incrementally updated with new data points. The buffer is
    used to train a classifier. The classifier is used to predict the labels
    of new data points. The new data points are then added to the buffer using
    the RECORDS algorithm, implemented in `RECORDS_choice`.

    A number of classification methods are supported:
        - MT (Mean Teacher) [1]
        - S3VM (Semi-supervised Support Vector Machine) [2]
        - LP (Label Propagation) [3]

    [0] Guo, L.-Z., Zhou, Z., & Li, Y.-F. (2020). RECORD: Resource constrained
      semi-supervised learning under distribution shift. In R. Gupta, Y. Liu,
      J. Tang, & B. A. Prakash (Eds.), KDD '20: The 26th ACM SIGKDD conference
      on knowledge discovery and data mining, virtual event, CA, USA, august
      23-27, 2020 (pp. 1636-1644). ACM. https://doi.org/10.1145/3394486.3403214

    [1] Antti Tarvainen and Harri Valpola. 2017. Mean teachers are better
      role models: Weight-averaged consistency targets improve
      semi-supervised deep learning results. In NeurIPS. 1195-1204.

    [2] Bennett, K., & Demiriz, A. (1998). Semi-supervised support vector
      machines. Advances in Neural Information processing systems, 11.

    [3] Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled
      data with label propagation. Technical Report CMU-CALD-02-107,
      Carnegie Mellon University, 2002
      http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf
    """

    def __init__(
        self,
        batch_size: int,
        buffer_budget: int,
        features: int,
        method: t.Literal["LP", "S3VM", "MT"],
        class_labels: t.List[int],
    ):
        """RECORD constructor

        :param batch_size: How often to update the classifier with a new batch or chunk
        :param buffer_budget: How many data points can be stored in the buffer
        :param features: The number of features in each data point
        :param method: The semi-supervised classification method to use
        :param class_labels: The possible class labels
        """
        super().__init__()
        self.unlabeled_chunk = ChunkQueue(batch_size)
        self.method = method
        self.buffer = RecordBuffer(buffer_budget, features)
        self.classifier = None
        self.class_labels = class_labels

    def _fit_label_propagation(self, unlabeled_x: np.ndarray) -> None:
        self.classifier = LabelPropagation(
            kernel="rbf", n_neighbors=9, max_iter=100000, n_jobs=-1
        )
        train_x = np.vstack([self.buffer.x_buffer, unlabeled_x])
        train_y = np.vstack([self.buffer.y_buffer, -np.ones((unlabeled_x.shape[0], 1))])
        self.classifier.fit(train_x, train_y.ravel())

    def _fit_s3vm(self, unlabeled_x: np.ndarray) -> None:
        self.classifier = S3VM(max_iter=300, kernel="rbf")
        self.classifier.fit(self.buffer.x_buffer, self.buffer.y_buffer, unlabeled_x)

    def _fit_mt(self, unlabeled_x: np.ndarray) -> None:
        self.classifier = MT(self.class_labels, unlabeled_x.shape[1])
        self.classifier.fit(
            self.buffer.x_buffer, unlabeled_x, self.buffer.y_buffer.ravel()
        )

    def _learn_from_chunk(self, x: np.ndarray) -> None:
        # Train
        if self.method == "LP":
            self._fit_label_propagation(x)
        elif self.method == "S3VM":
            self._fit_s3vm(x)
        elif self.method == "MT":
            self._fit_mt(x)

        # Update buffer
        predictions = self.classifier.predict(x)
        probabilities = self.classifier.predict_proba(x)
        self.buffer.update(x, predictions, probabilities)

    def learn_one(self, x: dict, y: SemiSupervisedLabel) -> SemiSupervisedClassifier:
        if y is not None:
            x, y = convert_river_to_skmf(x, y)
            self.buffer.add_labeled_instance(x, y)
            return self

        self.unlabeled_chunk.add(x, y)
        if self.unlabeled_chunk.chunk_ready():
            x, _ = self.unlabeled_chunk.take_chunk()
            self._learn_from_chunk(x)
        return self

    def is_classifier_ready(self) -> bool:
        """Early in the stream, the classifier may not be ready to make
        predictions. This method returns True if the classifier is ready to
        make predictions, False otherwise.
        """
        return self.classifier is not None

    def predict_proba_one(self, x: dict) -> dict:
        if self.classifier is None:
            raise ValueError("Classifier has not been trained yet")

        # Convert SciKit to river format
        probabilities = self.classifier.predict_proba(convert_river_to_skmf(x)[0])[0]
        probability_dict = {}
        for i, label in enumerate(self.class_labels):
            label = int(label)
            probability_dict[label] = probabilities[i]

        return probability_dict
