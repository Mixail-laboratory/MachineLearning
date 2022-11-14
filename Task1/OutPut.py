import numpy as np
from matplotlib import pyplot as plt


def show_data(X, y, feature_names, numbers_of_plot):
    if numbers_of_plot <= 1:
        raise "Numbers of features not enough"

    if numbers_of_plot == 2:
        fig, axs = plt.subplots(1)
        fig.suptitle(str(numbers_of_plot) + ' features visualization')

        axs.scatter(X[:, 0], X[:, 1], c=y, marker='.')
        axs.set_title(feature_names[0])
        axs.set_ylabel(feature_names[1])

    else:
        plt.figure(figsize=(15, 10))
        count = 1
        for i in range(numbers_of_plot):
            for j in range(i, numbers_of_plot):
                if i != j:
                    plt.subplot(numbers_of_plot, numbers_of_plot, count)
                    plt.scatter(X[:, i], X[:, j], c=y, marker='.')
                    plt.xlabel(feature_names[i])
                    plt.ylabel(feature_names[j])
                    count += 1

    plt.show()


def result_printer(model, true_data, data_to_predict):
    predictions = model.predict(data_to_predict)
    print("model accuracy:", np.sum(true_data == predictions) / len(predictions))
