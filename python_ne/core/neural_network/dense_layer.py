import random as rnd
import numpy as np
import copy


class DenseLayer:

    def __init__(self, units, activation, input_shape=None, weights=(None, None,)):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.weights, self.bias = weights

    def initialize(self):
        if self.weights is not None and self.bias is not None:
            return

        self.bias = np.array([rnd.uniform(-1, 1) for _ in range(self.units)])
        # number os neurons of the prev layer and number of neurons of this layer, shape = (input_shape[0], units)
        self.weights = np.array([[rnd.uniform(-1, 1) for j in range(self.units)] for i in range(self.input_shape[0])])

    def feedforward(self, xs):
        output = xs.dot(self.weights)
        return self.activation(output + self.bias)

    def get_weights(self):
        return copy.copy(self.weights), copy.copy(self.bias)

    def set_weights(self, weights):
        self.weights, self.bias = weights
