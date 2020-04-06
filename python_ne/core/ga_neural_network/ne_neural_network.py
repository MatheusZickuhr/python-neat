import random

import numpy as np
from python_ne.core.ga_neural_network.ga_neural_network import GaNeuralNetwork
from python_ne.core.ga import randomly_combine_lists
from python_ne.core.neural_network.dense_layer import DenseLayer
from python_ne.core.neural_network.neural_network import NeuralNetwork
from python_ne.core.neural_network import activations


class NeNeuralNetwork(GaNeuralNetwork):
    def create_model(self):
        model = NeuralNetwork()
        model.add(DenseLayer(activation=activations.sigmoid, input_shape=self.input_shape, units=128, ))
        model.add(DenseLayer(activation=activations.sigmoid, units=128, ))
        model.add(DenseLayer(activation=activations.sigmoid, units=self.output_size, ))
        return model

    def get_output(self, obs):
        return np.argmax(self.model.predict(obs))

    def crossover(self, other):
        return self.simple_crossover(other)

    def complex_crossover(self, other):
        child1 = NeNeuralNetwork(create_model=False, input_shape=self.input_shape, output_size=self.output_size)
        child2 = NeNeuralNetwork(create_model=False, input_shape=self.input_shape, output_size=self.output_size)

        child1.model = NeuralNetwork()
        child2.model = NeuralNetwork()

        for parent1_layer, parent2_layer in zip(self.model.layers, other.model.layers):
            parent1_weights = parent1_layer.get_weights()[0]
            parent2_weights = parent2_layer.get_weights()[0]

            parent1_bias = parent1_layer.get_weights()[1]
            parent2_bias = parent2_layer.get_weights()[1]

            prev_layer_neuron_count, current_layer_neuron_count = parent1_weights.shape

            total_weights_count = prev_layer_neuron_count * current_layer_neuron_count

            parent1_weights_flat = parent1_weights.reshape((total_weights_count,))
            parent2_weights_flat = parent2_weights.reshape((total_weights_count,))

            weight_combination1, weight_combination2 = randomly_combine_lists.get_random_lists_combinations(
                parent1_weights_flat, parent2_weights_flat, (prev_layer_neuron_count, current_layer_neuron_count))

            bias_combination_1, bias_combination_2 = randomly_combine_lists \
                .get_random_lists_combinations(parent1_bias, parent2_bias, return_shape=parent1_bias.shape)

            child1.model.add(DenseLayer(
                weights=[weight_combination1, bias_combination_1],
                input_shape=(parent1_layer.input_shape[-1],),
                units=parent1_layer.units,
                activation=activations.sigmoid
            ))
            child2.model.add(DenseLayer(
                weights=[weight_combination2, bias_combination_2],
                input_shape=(parent1_layer.input_shape[-1],),
                units=parent1_layer.units,
                activation=activations.sigmoid
            ))
        return child1, child2

    def simple_crossover(self, other):
        n_children = 2
        children = []

        for i in range(n_children):
            child = NeNeuralNetwork(create_model=False, input_shape=self.input_shape, output_size=self.output_size)
            children.append(child)
            child.model = NeuralNetwork()
            for layers in zip(self.model.layers, other.model.layers):
                chosen_layer = random.choice(layers)
                child.model.add(
                    DenseLayer(
                        units=chosen_layer.units,
                        input_shape=chosen_layer.input_shape,
                        weights=chosen_layer.get_weights(),
                        activation=activations.sigmoid
                    )
                )
        return children

    def mutate(self):
        for layer in self.model.layers:
            weights = layer.get_weights()[0]
            bias = layer.get_weights()[1]

            prev_layer_neuron_count, current_layer_neuron_count = weights.shape

            for prev_neuron in range(prev_layer_neuron_count):
                for cur_neuron in range(current_layer_neuron_count):
                    if random.random() < 0.1:
                        weights[prev_neuron][cur_neuron] = random.uniform(-1, 1)

            for i in range(len(bias)):
                if random.random() < 0.1:
                    bias[i] = random.uniform(-1, 1)

            layer.set_weights((weights, bias))
