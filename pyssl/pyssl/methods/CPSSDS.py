from typing import Dict, Literal

import numpy as np
from river.base.typing import ClfTarget

from river.tree import HoeffdingTreeClassifier
from river.naive_bayes import GaussianNB

from pyssl.skmf import (
    ChunkQueue,
    ClassifierBatched,
    convert_one_skmf_to_river,
    shuffle_split,
    split_by_label_presence,
)
from pyssl.ssl_classifier import SemiSupervisedClassifier
from pyssl.ssl_stream import SemiSupervisedLabel


def Unlabeling_data(X_train, Y_train, Percentage, chunk_size, class_count):
    labeled_count = round(Percentage * chunk_size)
    TLabeled = X_train[0 : labeled_count - 1]
    Y_TLabeled = Y_train[0 : labeled_count - 1]
    X_Unlabeled = X_train[labeled_count : Y_train.shape[0] - 1]

    cal_count = round(0.3 * TLabeled.shape[0])
    X_cal = TLabeled[0 : cal_count - 1]
    Y_cal = Y_TLabeled[0 : cal_count - 1]
    X_L = TLabeled[cal_count : TLabeled.shape[0] - 1]
    Y_L = Y_TLabeled[cal_count : TLabeled.shape[0] - 1]

    return X_Unlabeled, X_L, Y_L, X_cal, Y_cal


def Prediction_by_CP(num, classifier, X, Y, X_Unlabeled, class_count, sl):
    row = X_Unlabeled.shape[0]
    col = class_count
    p_values = np.zeros([row, col])
    labels = np.ones((row, col), dtype=bool)
    alphas = NCM(num, classifier, X, Y, 1, class_count)
    for elem in range(row):
        c = []
        for o in range(class_count):
            a_test = NCM(
                num, classifier, np.array([X_Unlabeled[elem, :]]), o, 2, class_count
            )
            idx = np.argwhere(Y == o).flatten()
            temp = alphas[idx]
            p = len(temp[temp >= a_test])
            if idx.shape[0] == 0:
                s = 0
            else:
                s = p / idx.shape[0]
            c.append(s)
            if s < sl:
                labels[elem, int(o)] = False
        p_values[elem, :] = np.array(c)
    return p_values, labels


def NCM(num, classifier, X, Y, t, class_count):
    if num == 1:
        if t == 1:
            p = np.zeros([X.shape[0], 1])
            alpha = np.zeros([X.shape[0], 1])
            for g in range(X.shape[0]):
                dic_vote = classifier.predict_proba_one(
                    convert_one_skmf_to_river([X[g, :]])
                )
                print(dic_vote)
                vote = np.fromiter(dic_vote.values(), dtype=float)
                vote_keys = np.fromiter(dic_vote.keys(), dtype=int)
                Sum = np.sum(vote)
                keys = np.argwhere(vote_keys == int(Y[g])).flatten()
                if keys.size == 0:
                    p[g] = (1) / (Sum + class_count)
                else:
                    for key, val in dic_vote.items():
                        if key == float(Y[g]):
                            p[g] = (val + 1) / (Sum + class_count)
                alpha[g] = 1 - p[g]

        else:
            dic_vote = classifier.predict_proba_one(convert_one_skmf_to_river(X[0, :]))
            vote = np.fromiter(dic_vote.values(), dtype=float)
            vote_keys = np.fromiter(dic_vote.keys(), dtype=int)
            Sum = np.sum(vote)
            keys = np.argwhere(vote_keys == int(Y)).flatten()
            if keys.size == 0:
                p = (1) / (Sum + class_count)
            else:
                for key, val in dic_vote.items():
                    if key == float(Y):
                        p = (val + 1) / (Sum + class_count)
            alpha = 1 - p

    else:
        if t == 1:
            prediction = classifier.predict_proba_many(X)
            P = np.max(prediction, axis=1)
            alpha = 1 - P
        elif t == 2:
            prediction = classifier.predict_proba_many(X)
            P = prediction[0, int(Y)]
            alpha = 1 - P
    return alpha


def Informatives_selection(X_Unlabeled, p_values, labels, class_count):
    row = X_Unlabeled.shape[0]
    X = np.empty([1, X_Unlabeled.shape[1]])
    Y = np.empty([1])
    for elem in range(row):
        l = np.argwhere(labels[elem, :] == True).flatten()
        if len(l) == 1:
            pp = p_values[elem, l]
            X = np.append(X, [X_Unlabeled[elem, :]], axis=0)
            Y = np.append(Y, [l[0]], axis=0)
    Informatives = X[1 : X.shape[0], :]
    Y_Informatives = Y[1 : Y.shape[0]]
    return Informatives, Y_Informatives


def Appending_informative_to_nextchunk(
    X_Currentchunk_Labeled, Y_Currentchunk_Labeled, Informatives, Y_Informatives
):
    X = np.append(X_Currentchunk_Labeled, Informatives, axis=0)
    Y = np.append(Y_Currentchunk_Labeled, Y_Informatives, axis=0)
    return X, Y


class CPSSDS(SemiSupervisedClassifier):
    """Conformal prediction for semi-supervised classification on data streams"""

    def __init__(
        self,
        base_model: Literal["NaiveBayes", "HoeffdingTree"],
        chunk_size: int = 500,
        class_count: int = 2,
        significance_level: float = 0.98,
    ) -> None:
        super().__init__()
        self.chunk_size: int = chunk_size
        self.significance_level: float = significance_level
        self.chunk_queue = ChunkQueue(chunk_size)
        self.chunk_id = 0
        self.class_count = class_count
        self.calibration_split = 0.3

        if base_model == "NaiveBayes":
            self.classifier = GaussianNB()
            self._num = 2
        elif base_model == "HoeffdingTree":
            self.classifier = HoeffdingTreeClassifier()
            self._num = 1
        else:
            raise ValueError("`base_model` must be either NaiveBayes or HoeffdingTree")

        self.classifier = ClassifierBatched(self.classifier)

        # Self-labeled data, initialized as empty
        self.self_labeled_x: np.array = None
        self.self_labeled_y: np.array = None

    def _learn_from_chunk(self, x: np.ndarray, y: np.ndarray) -> None:
        (x_label, y_label), x_unlabeled = split_by_label_presence(x, y)
        (x_cal, y_cal), (x_train, y_train) = shuffle_split(
            self.calibration_split, x_label, y_label
        )

        # Add self-labeled data to training data
        if self.self_labeled_x is not None:
            x_train = np.concatenate((x_train, self.self_labeled_x))
            y_train = np.concatenate((y_train, self.self_labeled_y))
        self.classifier.learn_many(x_train, y_train)

        # Use conformal prediction to label some unlabeled data
        p_values, labels = Prediction_by_CP(
            self._num,
            self.classifier,
            x_cal,
            y_cal,
            x_unlabeled,
            self.class_count,
            self.significance_level,
        )

        # Add newly labeled data to self-labeled data
        self.self_labeled_x, self.self_labeled_y = Informatives_selection(
            x_unlabeled, p_values, labels, self.class_count
        )

    def learn_one(self, x: dict, y: SemiSupervisedLabel) -> SemiSupervisedClassifier:
        self.chunk_queue.add(x, y)
        if self.chunk_queue.chunk_ready():
            x, y = self.chunk_queue.take_chunk()
            self._learn_from_chunk(x, y)
            self.chunk_id += 1
        return self

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        return self.classifier.predict_proba_one(x)
