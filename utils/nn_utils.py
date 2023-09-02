import numpy as np

def sigmoid(X):
    Y = 1 / (1+np.exp(-X))
    cache = X
    return Y, cache

def relu(X):
    Y = np.maximum(0, X)
    cache = X
    return Y, cache

def sigmoid_backward(dY, cache):
    X = cache
    Y = 1/(1+np.exp(-X))
    dX = dY * Y * (1-Y)
    return dX

def relu_backward(dY, cache):
    X = cache
    dX = np.array(dY, copy=True)
    dX[X <= 0] = 0
    return dX
    