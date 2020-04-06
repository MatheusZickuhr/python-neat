import random

import numpy as np
import copy


class DenseLayer:

    def __init__(self, units, activation, input_shape=None, weights=(None, None,)):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.weights, self.bias = weights
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        self.bias = np.array([random.uniform(-1, 1) for _ in range(self.units)])
        # number os neurons of the prev layer and number of neurons of this layer, shape = (input_shape[0], units)
        self.weights = np.array([[random.uniform(-1, 1) for j in range(self.units)] for i in range(self.input_shape[0])])
        self.initialized = True

    def feedforward(self, x_list):
        output = np.zeros(self.units)
        weighted_xs = np.zeros((len(x_list), self.units))

        i_len = len(x_list)
        j_len = self.units

        for i in range(i_len):
            for j in range(j_len):
                weighted_xs[i][j] = (
                        self.weights[i][j] * x_list[i]
                )

        for j in range(j_len):
            weighted_sum = 0
            for i in range(i_len):
                weighted_sum += weighted_xs[i][j]
            output[j] = self.activation(weighted_sum + self.bias[j])

        return output

    def get_weights(self):
        return copy.copy(self.weights), copy.copy(self.bias)

    def set_weights(self, weights):
        self.weights, self.bias = weights
