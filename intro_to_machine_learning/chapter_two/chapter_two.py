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
    plt.show()

def breast_cancer_dataset():
    cancer = load_breast_cancer()
    print(f'Cancer.keys: \n{cancer.keys()}')
    print(f'Shape of cancer data {cancer.data.shape}')
    print('Sample counts per class: {}'.format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    ))
    print(f'Feature names: {cancer.feature_names}')

def boston_housing_dataset():
    boston = load_boston()
    print(f'Data shape: {boston.data.shape}')
    X, y = mglearn.datasets.load_extended_boston()
    print(f'X shape: {X.shape}')

boston_housing_dataset()