import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data():
    train_dataset = h5py.File('../data/train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])
    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))

    test_dataset = h5py.File('../data/test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])
    test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

    classes = np.array(test_dataset["list_classes"][:])
    return train_x_orig, train_y, test_x_orig, test_y, classes

def preprocess_data(train_x_orig, test_x_orig):
    # flatten the image data
    train_x_flat = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flat = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # standardize the data
    train_x = train_x_flat / 255
    test_x = test_x_flat / 255
    return train_x, test_x

def load_planar_data():
    np.random.seed(1)
    size = 400 # number of examples
    N = int(size/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((size,D)) # data matrix where each row is a single example
    Y = np.zeros((size,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y