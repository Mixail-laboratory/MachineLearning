from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn import metrics

from Distance import *
from Kernel import *


class PotentialKNeighborsClassifier(BaseEstimator):
    def __init__(self, window_width, epoch_number) -> None:
        self.classes = None  # np.unique(train_y)
        self.gammas = None  # np.zeros_like(train_y)
        self.indexes = None  # np.arange(0, len(train_y))
        self.Kernel = potential_kernel
        self.window_width = window_width
        self.epoch_number = epoch_number

        self.train_x = None
        self.train_y = None

        self.zero_x = None
        self.zero_y = None

    def predict(self, x: np.array):
        test_x = np.copy(x)

        if len(test_x.shape) < 2:
            test_x = test_x[np.newaxis, :]

        u = test_x[:, np.newaxis, :]
        v = self.train_x[np.newaxis, :, :]
        weights = self.gammas * self.Kernel(euclidean_distance(u, v) / self.window_width)
        scores = np.vstack(
            [np.sum(weights.T[np.where(self.train_y == 0)[0]].T, axis=1),
             np.sum(weights.T[np.where(self.train_y == 1)[0]].T, axis=1),
             np.sum(weights.T[np.where(self.train_y == 2)[0]].T, axis=1)]).T

        return np.argmax(scores, axis=1)

    def fit(self, train_x, train_y):
        self.classes = np.unique(train_y)
        self.gammas = np.zeros_like(train_y)
        self.train_x = np.copy(train_x)
        self.train_y = np.copy(train_y)

        self.indexes = np.arange(0, len(train_y))

        for _ in range(self.epoch_number):
            for i in range(self.train_x.shape[0]):
                if self.predict(self.train_x[i]) != self.train_y[i]:
                    self.gammas[i] += 1

        # get samples with zero potentials
        zero_mask = self.gammas == 0
        self.zero_x = self.train_x[zero_mask, ...]
        self.zero_y = self.train_y[zero_mask, ...]
        self.indexes = self.indexes[zero_mask, ...]

    def get_bad_prediction_arr(self, test_x, test_y):
        bad_predictions_array = list()
        predict_arr = self.predict(test_x)
        for i in range(len(test_y)):
            if predict_arr[i] != test_y[i]:
                bad_predictions_array.append(i)
        return bad_predictions_array

    def show_accuracy(self, X, y, test_x, test_y):
        predict_arr = self.predict(test_x)
        print("Accuracy")
        print("On test  = ", metrics.accuracy_score(test_y, predict_arr))
        print("On train = ", metrics.accuracy_score(self.train_y, self.predict(self.train_x)))
        print("On full data: ", metrics.accuracy_score(y, self.predict(X)))
