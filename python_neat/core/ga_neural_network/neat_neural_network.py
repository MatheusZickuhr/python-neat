import random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from python_neat.core.ga_neural_network.neural_network import NeuralNetwork

number_of_neurons_choices = [16, 32, 64, 128, 256, 512, 1024]


class NeatNeuralNetwork(NeuralNetwork):

    def __init__(self, topology=None, *args, **kwargs):
        super(NeatNeuralNetwork, self).__init__(*args, **kwargs)
        self.topology = self.create_topology() if topology is None else topology

    def create_topology(self):
        hidden_layers_count = random.randint(2, 8)
        return [random.choice(number_of_neurons_choices) for _ in range(hidden_layers_count)]

    def create_model(self):
        model = Sequential()

        for i, units in enumerate(self.topology):
            model.add(Dense(units, input_shape=self.input_shape, ) if i == 0 else Dense(units, ))

        model.add(Dense(self.output_size, activation='softmax'))

        return model

    def get_output(self, obs):
        return np.argmax(self.model.predict(obs))

    def crossover(self, other):
        child1 = NeatNeuralNetwork(
            input_shape=self.input_shape,
            output_size=self.output_size,
            topology=[*self.topology[:len(self.topology) // 2], *other.topology[len(other.topology) // 2:]]
        )

        child2 = NeatNeuralNetwork(
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
                        weights[i] = random.uniform(-1, 1)
                else:
                    mutate_weights(weights[i])

        for layer in self.model.layers:
            new_weights = layer.get_weights()
            mutate_weights(new_weights)
            layer.set_weights(new_weights)

    def __str__(self):
        return f'fitness = {self.fitness}'
