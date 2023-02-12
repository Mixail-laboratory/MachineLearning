from sklearn.base import BaseEstimator
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

        self.zero_weights = None
        self.valid_indixes = None
        

    def predict(self, x: np.array):
        test_x = np.copy(x)

        if len(test_x.shape) < 2:
            test_x = test_x[np.newaxis, :]
        self.valid_indixes = np.arange(0, len(test_x))    

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
        non_zero_mask = self.gammas != 0
        self.indexes = self.indexes[non_zero_mask, ...]

    def show_accuracy(self, X, y, test_x, test_y):
        predict_arr = self.predict(test_x)
        print("Accuracy")
        print("On test  = ", metrics.accuracy_score(test_y, predict_arr))
        print("On train = ", metrics.accuracy_score(self.train_y, self.predict(self.train_x)))
        print("On full data: ", metrics.accuracy_score(y, self.predict(X)))
