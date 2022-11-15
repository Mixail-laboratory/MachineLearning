import queue

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

from Distance import *
from Kernel import *


class PotentialKNeighborsClassifier(BaseEstimator):
    def __init__(self, distance=euclidean_distance, kernel=potential_kernel, window_width=1, max_iteration=5000):
        self.distance = distance
        self.kernel = kernel
        self.gamma = None
        self.gamma_no_modified = None
        self.window_width = window_width
        self.n_classes = 0
        self.class_index_array = list()
        self.X_train = None
        self.label_for_visualize = None
        self.label_for_visualize_no_modified = None
        self.X_train_no_modified = None
        self.class_index_array_no_modified = None
        self.max_iteration = max_iteration

    def __get_prediction_for_element__(self, elem):
        type_of_prediction = np.zeros(len(self.class_index_array))

        for i in range(0, len(self.class_index_array)):
            distance = self.distance(elem, self.X_train[self.class_index_array[i]])
            type_of_prediction[i] = (self.kernel(distance / self.window_width) *
                                     self.gamma[self.class_index_array[i]]).sum()
        return np.argmax(type_of_prediction)

    def __init_index_of_classes__(self, y_train):
        tmp_index_array = list()
        for i in range(0, self.n_classes):
            tmp_index_array.append(list())

        for i in range(0, len(y_train)):
            tmp_index_array[int(y_train[i])].append(int(i))
        for i in range(0, self.n_classes):
            tmp_index_array[i] = np.asarray(tmp_index_array[i], dtype=np.int64)
        return tmp_index_array

    def __train_gamma__(self, y_train):
        total_number = len(self.X_train)

        window_size = 5
        eps = 0.2  # mean is the same with acc = 0.2
        iteration_count = 0
        q1 = queue.Queue(window_size)

        while iteration_count < self.max_iteration:
            cur_error_number = 0
            for i in range(0, total_number):
                prediction = self.__get_prediction_for_element__(self.X_train[i])
                if prediction != y_train[i]:
                    self.gamma[i] += 1
                    cur_error_number += 1

            if not q1.full():
                q1.put(cur_error_number)
            else:
                tmp = q1.get()
                q1.put(cur_error_number)
                if abs(tmp - cur_error_number) < eps * window_size:
                    # print("Learned %", 1.0 - cur_error_number / total_number)
                    # print("Gamma calculation took:", iteration_count)
                    return 1.0 - cur_error_number / total_number
            iteration_count += 1
            # Window size 3: a b c;
            # after 3 times sum=a+b+c
            # 4 times: sum1 = a+b+c - a + d
            # check if less than eps: sum1/len(sum1)-sum/len(sum1) < eps
            # d-a < eps * len(sum1)
            # 5 times: sum2 = b+c+d - b + e

    def __form_non_zero_gamma_objects__(self, y_train):

        non_zero_gamma_index = list()

        for i in range(0, len(self.gamma)):
            if int(self.gamma[i]) != 0:
                non_zero_gamma_index.append(i)

        non_zero_gamma_index = np.asarray(non_zero_gamma_index, dtype=np.int64)

        self.X_train_no_modified = self.X_train

        self.gamma_no_modified = self.gamma
        self.gamma = self.gamma[non_zero_gamma_index]
        self.X_train = self.X_train[non_zero_gamma_index]

        self.label_for_visualize = y_train[non_zero_gamma_index]
        self.label_for_visualize_no_modified = y_train

        self.class_index_array_no_modified = self.class_index_array
        self.class_index_array = self.__init_index_of_classes__(self.label_for_visualize)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.gamma = np.zeros(len(y))
        self.n_classes = sorted(y)[len(y) - 1] + 1
        self.X_train = X

        self.class_index_array = self.__init_index_of_classes__(y)
        trained_rate = self.__train_gamma__(y)

        self.__form_non_zero_gamma_objects__(y)
        return trained_rate

    def predict(self, X_test):
        predictions = list()
        for i in range(0, len(X_test)):
            predictions.append(self.__get_prediction_for_element__(X_test[i]))

        return np.array(predictions)

    # show zero potentials
    def predict_with_visualizer_no_modified(self, X_test, y_test, count_of_features, feature_names):
        predictions = list()
        plt.figure(figsize=(15, 10))
        count = 1
        for i in range(count_of_features):
            for j in range(i, count_of_features):
                plt.subplot(count_of_features, count_of_features, count)
                plt.scatter(self.X_train_no_modified[:, i], self.X_train_no_modified[:, j],
                            c='r', marker='x', s=15)
                plt.xlabel(feature_names[i])
                plt.ylabel(feature_names[j])

                correct_array = list()
                correct_index = list()
                for item in range(0, len(self.gamma_no_modified)):
                    if int(self.gamma_no_modified[item]) == 0:
                        correct_array.append(X_test[item])
                        correct_index.append(y_test[item])

                correct_array = np.asarray(correct_array, dtype=np.float64)

                if correct_array.size != 0:
                    plt.scatter(correct_array[:, i], correct_array[:, j],
                                c=np.asarray(correct_index, dtype=np.int64), marker='.', s=90)
                count += 1

        plt.show()
        return np.array(predictions)
