import random

import keras
import numpy as np
from keras.constraints import MinMaxNorm
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

activation_functions = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                        'exponential', 'linear']


class NeatAnn:

    def __init__(self, create_model=True, output_size=None, input_shape=None, topology=None):
        self.topology = self.create_topology() if topology is None else topology
        self.input_shape = input_shape
        self.output_size = output_size
        self.fitness = 0
        self.dense_layers = list()
        self.model = self.create_model() if create_model else None
        self.was_evaluated = False

    def create_topology(self):
        l = random.randint(2, 6)
        topology = []
        for i in range(l):
            topology.append(random.randint(16, 256))
        return topology

    def create_model(self):
        model = Sequential()

        for i, units in enumerate(self.topology):
            if i == 0:
                model.add(
                    Dense(
                        units, input_shape=self.input_shape,
                        kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                        bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
                    )
                )
            else:
                model.add(
                    Dense(
                        units,
                        kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                        bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
                    )
                )


        model.add(
            Dense(
                self.output_size, activation='linear', bias_initializer='random_normal',
                kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0),
                bias_constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
            )
        )

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def get_output(self, obs):
        return np.argmax(self.model.predict(obs))

    def crossover(self, other):
        child1 = NeatAnn(
            input_shape=self.input_shape,
            output_size=self.output_size,
            topology=[*self.topology[:len(self.topology) // 2], *other.topology[len(other.topology) // 2:]]
        )

        child2 = NeatAnn(
            input_shape=self.input_shape,
            output_size=self.output_size,
            topology=[*other.topology[:len(other.topology) // 2], *self.topology[len(self.topology) // 2:]]
        )
        return child1, child2

    def mutate(self):
        def mutate_weights(weights):
            for i in range(len(weights)):
                if type(weights[i]) == np.float32:
                    if random.random() < 0.20:
                        weights[i] = random.random()
                else:
                    mutate_weights(weights[i])

        for layer in self.model.layers:
            new_weights = layer.get_weights()
            mutate_weights(new_weights)
            layer.set_weights(new_weights)