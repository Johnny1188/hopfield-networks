import numpy as np
from itertools import islice

MNIST_PATH = "data/mnist_train.csv"

def get_data(max_rows=10000):
    data = None
    if max_rows:
        with open(MNIST_PATH, "rb") as f:
            data = np.genfromtxt(islice(f, 1, max_rows+1), delimiter=',')
    else:
        data = np.genfromtxt(MNIST_PATH, delimiter=',')[1:]
    X, y = data[:,1:], data[:,0]
    return X, y

def get_samples_per_class(X, y):
    uniq_classes = np.unique(y)
    sampled_X = np.zeros((uniq_classes.shape[0], X.shape[1]))
    for c_i, c in enumerate(uniq_classes):
        samples_of_c = X[y == c]
        if samples_of_c.shape[0] == 0:
            print(f"Class {c} not present.")
            continue
        sampled_X[c_i] = samples_of_c[0]
    return sampled_X, uniq_classes
