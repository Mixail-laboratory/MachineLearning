import numpy as np
from sklearn.tree import DecisionTreeClassifier

class MyRandomForest:
    def __init__(self, nums_of_trees, subset_size = 0.5, random_state = 0) -> None:
        self.classifiers = [DecisionTreeClassifier(criterion='gini', max_features='sqrt') for _ in range(nums_of_trees)]
        self.subset_size = subset_size
        self.random_state = random_state

    def fit(self, x_train : np.ndarray, y_train : np.ndarray):
        generate = np.random.RandomState(self.random_state)
        subset_size = int(x_train.shape[0] * self.subset_size)

        for classifier in self.classifiers:
            inidices = generate.choice(x_train.shape[0], subset_size)
            x_subset = x_train[inidices, ...]
            y_subset = y_train[inidices, ...]
            classifier.fit(x_subset, y_subset)
    
    def predict(self, x_test : np.ndarray):
        predictions = np.zeros((x_test.shape[0], len(self.classifiers)), dtype=np.int64)
        for i, cls in enumerate(self.classifiers):
            predictions[..., i] = cls.predict(x_test)
        result_pred = np.zeros((x_test.shape[0]), dtype=np.int64)

        for i, pred in enumerate(predictions):
            clases, counts = np.unique(pred, return_counts=True)
            result_pred[i] = clases[np.argmax(counts)]

        return result_pred