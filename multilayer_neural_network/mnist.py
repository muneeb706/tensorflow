# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:12:07 2018

@author: Hantao
"""

""" 
    Each element in train_data is (x, y), 
    where the shape of x is (784) and the shape of y is (10).  
    Each element in test_data or valid_data is (x, y), 
    where the shape of x is (784) and y is a digit.
    Each x above represents 28 * 28 = 784 pixels in an MNIST image.
"""
# Standard library
import pickle
import gzip
import numpy as np


def vectorized_digit(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    in (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10))
    e[j] = 1.0
    return e


""" The MNIST data stored in the zipped pickle file mnist.pkl.gz is 
    a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a pair of two entries:
    The first entry is a numpy ndarray of 50,000 actual training images.  
    Each image is a numpy ndarray with 784 values, 
    representing the 28 * 28 = 784 pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
"""
f = gzip.open('mnist.pkl.gz', 'rb')
train_data, valid_data, test_data = pickle.load(f, encoding="latin1")
f.close()

""" Draw one digit """
from matplotlib import pyplot as plt
import random

r = random.randint(0, 100)
x = valid_data[0][r]
y = valid_data[1][r]
plt.imshow(x.reshape(28, 28))
plt.show()
print("The {}th digit is {}.".format(r + 1, y))

train_X = train_data[0]
train_Y = [vectorized_digit(y) for y in train_data[1]]

import time

start_time = time.time()

import mlnn

# net = mlnn.Network([784, 60, 30, 10]) # eta=0.04
net = mlnn.Network([784, 30, 10])
print('Network =', net.sizes)
print('weight shapes:', [w.shape for w in net.weights])
net.SGD(train_X, train_Y, epochs=10,
        batch_size=100, eta=0.06, test_data=test_data)
# For Cross-entropy, try small learning rates.

print("run time: %s seconds" % (time.time() - start_time))