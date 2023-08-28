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

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)