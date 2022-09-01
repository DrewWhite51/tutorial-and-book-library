import numpy as np
from sklearn.datasets import load_iris
import pprint
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor

def first_dataset():
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['Class 0', 'Class 1'], loc=4)
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    print(f'X.shape {X.shape}')
    plt.show()

def wave_dataset():
    X, y = mglearn.datasets.make_wave(n_samples = 40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel('Feature')
    plt.ylabel('Target')
    #plt.show()
    mglearn.plots.plot_knn_regression(n_neighbors=4)
    plt.show()


def breast_cancer_dataset():

    cancer = load_breast_cancer()
    print(f'Cancer.keys: \n{cancer.keys()}')
    print(f'Shape of cancer data {cancer.data.shape}')
    print('Sample counts per class: {}'.format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    ))
    print(f'Feature names: {cancer.feature_names}')
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66)
    training_accuracy = []
    test_accuracy = []
    # try n_neighbors from 1 to 10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # build the model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        # record training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
        # record generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))
    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()



def boston_housing_dataset():

    boston = load_boston()
    print(f'Data shape: {boston.data.shape}')
    X, y = mglearn.datasets.load_extended_boston()
    print(f'X shape: {X.shape}')
    mglearn.plots.plot_knn_classification(n_neighbors=5)
    plt.show()

def k_nearest_algo():

    X, y = mglearn.datasets.make_forge()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X_train, y_train)
    print(f'Test set predictions {clf.predict(X_test)}')
    print(f"Test set accuracy: {clf.score(X_test, y_test)}")
    # The code below produces a decision boundary which is where the algorithm assigns
    # class 0 vs. class 1.
    fig, axes = plt.subplots(1, 3, figsize=(10,3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.show()

def k_nearest_regression_algo():

    X, y = mglearn.datasets.make_wave(n_samples=40)
    # Split to training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Instantiate the model and set the number of neighbors to consider 3
    reg = KNeighborsRegressor(n_neighbors = 3)
    # Fitting the model using the training data and training targets
    reg.fit(X_train, y_train)
    print(f"Test set predictions:{reg.predict(X_test)}")
    print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

k_nearest_regression_algo()