import numpy as np
from sklearn.datasets import load_iris
import pprint
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

# print(type(iris_dataset['data']))
# print(iris_dataset['data'].shape)
# Shows the first 5 rows of the data
# print(iris_dataset['data'][:5])

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Printing the shape of train and test data.
# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))
#
# print("X_test shape: {}".format(X_test.shape))
# print("y_test shape: {}".format(y_test.shape))

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)


def graph_data():
    """
    This graph shows a good separation between all the data, which means we could design an algorithm that
    can easily detect the differences between the flowers.
    :return:
    """
    pd.plotting.scatter_matrix(iris_dataframe, figsize=(20, 20), grid=True,
                               marker='o', c=y_train, hist_kwds={'bins': 20}, s=60, alpha=.8)
    # iris_dataframe.scatter_matrix()
    plt.show()


# graph_data()

def k_nearest_classification():
    knn = KNeighborsClassifier(
        n_neighbors=1)  # Encapsulates algorithm that will be used to build the model from the training data,

    knn.fit(X_train, y_train)  # Building model on training set.

    X_new = np.array([[5, 2.9, 1, 0.2]])

    print(f'X_new.shape: {X_new.shape}')  # Important to note, the dimensions of the array.

    prediction = knn.predict(X_new)
    print(f'Prediction: {prediction}')
    print(f'Predicted target name: {iris_dataset["target_names"][prediction]}')

    y_pred = knn.predict(X_test)
    print(f'Testing predictions for: \n{y_pred}')
    # Both of these test the accuracy of the model
    print(f'Test set score: {np.mean(y_pred == y_test)}')
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


k_nearest_classification()
