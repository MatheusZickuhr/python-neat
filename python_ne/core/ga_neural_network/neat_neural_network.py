import random
import numpy as np
from python_ne.core.ga_neural_network.ga_neural_network import GaNeuralNetwork


class NeatNeuralNetwork(GaNeuralNetwork):

    def create_model(self):
        model = super().create_model()

        model.initialize()

        # randomly disable some neurons to create different topologies
        for layer in model.get_layers():
            layer_weights = layer.get_weights()
            weights = layer_weights[0]
            bias = layer_weights[1]

            for j in range(len(weights[0])):
                if random.random() < 0.1:
                    for i in range(len(weights)):
                        weights[i][j] = 0

                    bias[j] = 0

            layer.set_weights((weights, bias))

        return model

    def crossover(self, other):
        n_children = 2
        children = []

        for i in range(n_children):
            child = NeatNeuralNetwork(
                create_model=False,
                model_adapter=self.model_adapter,
                neural_network_config=self.neural_network_config
            )
            children.append(child)
            child.model = self.model_adapter()
            for layers in zip(self.model.get_layers(), other.model.get_layers()):
                chosen_layer = random.choice(layers)
                child.model.add_dense_layer(
                    units=chosen_layer.get_units(),
                    input_shape=chosen_layer.get_input_shape(),
                    weights=chosen_layer.get_weights(),
                    activation=chosen_layer.get_activation()
                )

        return children

    def __str__(self):
        return f'fitness = {self.fitness}'
