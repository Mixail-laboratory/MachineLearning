from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from Classifier import PotentialKNeighborsClassifier
from Distance import *
from Kernel import *
from OutPut import *

iris_ds = datasets.load_iris()
numbers_of_features = 4
start_feature = 0
X = iris_ds.data[:, start_feature:numbers_of_features]
y = iris_ds.target
feature_names = iris_ds.feature_names

show_data(X, y, feature_names, numbers_of_features)

model = PotentialKNeighborsClassifier(distance=euclidean_distance, kernel=potential_kernel, window_width=1)

X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.2, random_state=0)
kF = KFold(n_splits=5, shuffle=True, random_state=1)

X = X_train_set
y = y_train_set
scores = []

with open("cross_validation_results.txt", 'w') as file:
    for train_index, test_index in kF.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(np.sum(y_test == predictions) / len(predictions))
        file.write("The cross-validation scores using custom method are \n{0}".format(scores))
        file.write('\n')
        file.write("Mean of k-fold scores using custom method is {0}".format(np.mean(scores)))
        file.write('\n')
        file.write('*' * 80 + '\n')

X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.2, random_state=0)

model.fit(X_train_set, y_train_set)

result_printer(model, y_test_set, X_test_set)

model.predict_with_visualizer_no_modified(X_train_set, y_train_set, numbers_of_features, feature_names)
# * - zero potentials
