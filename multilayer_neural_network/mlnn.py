# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:14:07 2018

@author: Hantao Zhang
"""

# 

"""
mlnn.py
~~~~~~~~~~

A module to implement the fully connected, multi-layer neural network.
Gradients are calculated by composition rules. Weights are updated
according to mini-batch stochastic gradient descent using backpropagation.  
Effort has been focused on making the code simple, easily readable, 
and easily modifiable.  It is not optimized, and omits many desirable 
features.
"""

# Standard libraries
# import random
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Third-party libraries
import numpy as np

#### Miscellaneous functions
def square_cost(actual_output, y):
    """Return the sum of square cost, where y is the desired output. """
    return np.sum((actual_output-y)**2)/2

def square_cost_derivative(actual_output, y):
    """Return the vector of partial derivatives for the actual output."""
    return (actual_output - y)

def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x, y):
    """Derivative of the sigmoid function."""
    return y*(1-y)

def tanh(x): 
    """ the hyperbolic tangent function """    
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_prime(x, y):     
    """ the derivative of hyperbolic tangent function """    
    return (1 + y)*(1 - y)

class Network(object):

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the sizes
        are [2, 3, 2] then it defines a two-layer network, with two inputs,
        followed by the first layer containing 3 neurons, the second layer 
        (i.e., the output layer) has 2 neurons.  
        The biases and weights for the network are initialized randomly, 
        using a Gaussian distribution with mean 0, and variance 1.  
        """
        np.random.seed(60)    # for reproducibility               
        self.activation = sigmoid         
        self.activation_derivative = sigmoid_prime 
        # self.activation = tanh         
        # self.activation_derivative = tanh_prime 
        self.cost_derivative = square_cost_derivative
        self.num_layers = len(sizes)-1
        self.sizes = sizes
        self.weights = [np.random.randn(x, y)
                                 for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = np.asarray(self.weights)
        self.biases = np.asarray([np.random.randn(y) for y in sizes[1:]])
        
    def feedforward(self, a):
        """ 
        Suppose network takes p inputs and produces q outputs
        Require: the last dimension of a is p.
        Return the output of the network, whose last dimension is q.
        """
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(a, w) + b)
        return a

    def SGD(self, train_X, train_Y, epochs, batch_size, eta,
            test_data=None):
        """
        Suppose network takes p inputs and produces q outputs
        Train the neural network using mini-batch stochastic
        gradient descent.  
        train_X represents the training inputs and train_Y the desired
        outputs, where train_X's shape is (n, p) and train_Y's shape is (n, q). 
        The other non-optional parameters are self-explanatory.  
        If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        n = len(train_X)
        if test_data: n_test = len(test_data[0])
        
            
        for j in range(epochs):
            if (batch_size > 0):
                train_X, train_Y = shuffle(train_X, train_Y)
                batches_X = [train_X[k:k+batch_size] 
                                for k in range(0, n, batch_size)]        
                batches_Y = [train_Y[k:k+batch_size] 
                                for k in range(0, n, batch_size)]
            
            for batch_X, batch_Y in zip(batches_X, batches_Y):
                self.update_batch(batch_X, batch_Y, eta)
                
            if test_data:
                print("Epoch {} : {} / {}".format(j, 
                      self.evaluate(test_data), n_test));
 
        if not test_data:
            print("Epoch {} complete".format(epochs))

    def update_batch(self, batch_X, batch_Y, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The batch_X is a list of input x and batch_Y is a list of 
        corresponding desired output y (i.e. labels), and ``eta``
        is the learning rate."""
        nabla_w, nabla_b = self.backprop(batch_X, batch_Y)
        for i in range(self.num_layers):
            self.weights[i] -= eta*np.sum(nabla_w[i], axis=0)
            self.biases[i] -= eta*np.sum(nabla_b[i], axis=0)
        
    def backprop(self, x, y):
        """(x, y) are a batch of examples.
        Return a tuple ``(nabla_b, nabla_w)`` representing 
        the gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists (batch) of np arrays, 
        similar to ``self.biases`` and ``self.weights``."""
        
        # feedforward
        y_act = x
        y_acts = [x] # list to store all the activations, layer by layer
        z_wsums = [] # list to store all the weighted sum vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(y_act, w) + b
            z_wsums.append(z)
            y_act = self.activation(z)
            y_acts.append(y_act)

        # place holder for all layers of weights and biases
        nabla_b = list(range(self.num_layers))
        nabla_w = list(range(self.num_layers))
        
        # backward propagation
        theta = self.cost_derivative(y_act, y)
        for i in range(self.num_layers-1, -1, -1):  
            ad = self.activation_derivative(z_wsums[i], y_acts[i+1])
            delta = np.multiply(theta, ad)                          
            y_hat = y_acts[i][:, :, np.newaxis]
            delta_hat = delta[:, np.newaxis, :]
            nabla_w[i] = np.multiply(y_hat, delta_hat)                
            nabla_b[i] = delta
            if (i>0): theta = np.dot(delta, np.transpose(self.weights[i]))
        
        return (nabla_w, nabla_b)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the incorrect result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        digits = np.argmax(self.feedforward(test_data[0]), axis=1)
        return np.count_nonzero(digits - test_data[1])

    def plot_decision_regions(self, X, y, points=200):
        markers = ('o', '^')
        colors = ('red', 'blue')
        cmap = ListedColormap(colors)
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        resolution = max(x1_max - x1_min, x2_max - x2_min)/float(points)
        #resolution = 0.01
     
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        input = np.array([xx1.ravel(), xx2.ravel()]).T 
        Z = np.empty(0)
        for i in range(input.shape[0]):
            val = self.feedforward(np.array(input[i]))
            if val < 0.5: val = 0 
            if val >= 0.5: val = 1
            Z = np.append(Z, val)

        Z = Z.reshape(xx1.shape)
        
        plt.pcolormesh(xx1, xx2, Z, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot all samples

        classes = ["False", "True"]
        for idx, cl in enumerate(np.unique(y[:0])):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=1.0, c=cmap(idx), marker=markers[idx], s=80, label=classes[idx])
            
        plt.xlabel('x-axis')            
        plt.ylabel('y-axis')
        plt.legend(loc='upper left')
        plt.show()          

if __name__ == '__main__':

    train_X = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    train_Y = np.expand_dims(np.array([0, 1, 1, 0]), axis=1)
    
    net = Network([2, 3, 4, 1])
    print('weight shapes:', [w.shape for w in net.weights])
    print('weights:', [w for w in net.weights])
        
    net.SGD(train_X, train_Y, 200, 1, 17)
    
    #for a in train_X: print(a, net.feedforward(a))
    for x, y in zip(train_X, net.feedforward(train_X)): print(x, y)
        
    net.plot_decision_regions(train_X, train_Y[:0]) # for binary input only