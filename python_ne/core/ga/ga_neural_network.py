import random
import numpy as np

from python_ne.core.ga import randomly_combine_lists


class GaNeuralNetwork:

    def __init__(self, model_adapter, neural_network_config=None, create_model=True):
        self.model_adapter = model_adapter
        self.neural_network_config = neural_network_config
        self.fitness = 0
        self.raw_fitness = 0
        self.dense_layers = list()
        self.model = self.create_model() if create_model else None
        self.was_evaluated = False

    def create_model(self):
        model = self.model_adapter()
        for i, layer_config in enumerate(self.neural_network_config):
            if i == 0:
                input_shape, unit_count, activation = layer_config
                model.add_dense_layer(activation=activation, input_shape=input_shape, units=unit_count, )
            else:
                unit_count, activation = layer_config
                model.add_dense_layer(activation=activation, units=unit_count, )
        return model

    def get_output(self, obs):
        return self.model.predict(obs)

    def mutate(self, mutation_chance):
        for layer in self.model.get_layers():
            weights, bias = layer.get_weights()

            prev_layer_neuron_count, current_layer_neuron_count = weights.shape

            for prev_neuron in range(prev_layer_neuron_count):
                for cur_neuron in range(current_layer_neuron_count):
                    if random.random() < mutation_chance:
                        weights[prev_neuron][cur_neuron] = random.uniform(-1, 1)

            for i in range(len(bias)):
                if random.random() < mutation_chance:
                    bias[i] = random.uniform(-1, 1)

            layer.set_weights((weights, bias))

    def crossover(self, other):
        return self.complex_crossover_new(other)

    def complex_crossover_new(self, other):
        n_children = 2
        children = []

        for _ in range(n_children):
            child = GaNeuralNetwork(create_model=False, model_adapter=self.model_adapter,
                                    neural_network_config=self.neural_network_config)
            children.append(child)
            child.model = self.model_adapter()
            for layer1, layer2 in zip(self.model.get_layers(), other.model.get_layers()):
                weights1, bias1 = layer1.get_weights()
                weights2, bias2 = layer2.get_weights()

                child_weights = np.zeros(weights1.shape)
                for i in range(weights1.shape[0]):
                    for j in range(weights1.shape[1]):
                        child_weights[i][j] = weights1[i][j] if random.random() < 0.5 else weights2[i][j]

                child_bias = np.zeros(bias1.shape)
                for i in range(bias1.shape[0]):
                    child_bias[i] = bias1[i] if random.random() < 0.5 else bias2[i]

                child.model.add_dense_layer(
                    weights=(child_weights, child_bias),
                    input_shape=layer1.get_input_shape(),
                    units=layer1.get_units(),
                    activation=layer1.get_activation()
                )

        return children

    def complex_crossover(self, other):
        child1 = GaNeuralNetwork(
            create_model=False,
            model_adapter=self.model_adapter,
            neural_network_config=self.neural_network_config
        )
        child2 = GaNeuralNetwork(
            create_model=False,
            model_adapter=self.model_adapter,
            neural_network_config=self.neural_network_config
        )

        child1.model = self.model_adapter()
        child2.model = self.model_adapter()

        for parent1_layer, parent2_layer in zip(self.model.get_layers(), other.model.get_layers()):
            parent1_weights, parent1_bias = parent1_layer.get_weights()
            parent2_weights, parent2_bias = parent2_layer.get_weights()

            prev_layer_neuron_count, current_layer_neuron_count = parent1_weights.shape

            total_weights_count = prev_layer_neuron_count * current_layer_neuron_count

            parent1_weights_flat = parent1_weights.reshape((total_weights_count,))
            parent2_weights_flat = parent2_weights.reshape((total_weights_count,))

            weight_combination1, weight_combination2 = randomly_combine_lists.get_random_lists_combinations(
                parent1_weights_flat, parent2_weights_flat, (prev_layer_neuron_count, current_layer_neuron_count))

            bias_combination_1, bias_combination_2 = randomly_combine_lists \
                .get_random_lists_combinations(parent1_bias, parent2_bias, return_shape=parent1_bias.shape)

            child1.model.add_dense_layer(
                weights=[weight_combination1, bias_combination_1],
                input_shape=parent1_layer.get_input_shape(),
                units=parent1_layer.get_units(),
                activation=parent1_layer.get_activation()
            )
            child2.model.add_dense_layer(
                weights=[weight_combination2, bias_combination_2],
                input_shape=parent1_layer.get_input_shape(),
                units=parent1_layer.get_units(),
                activation=parent1_layer.get_activation()
            )
        return child1, child2

    def simple_crossover(self, other):
        n_children = 2
        children = []

        for i in range(n_children):
            child = GaNeuralNetwork(
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

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        # load is an static method in the adapter class
        self.model = self.model_adapter.load(file_path)

    def __str__(self):
        return f'fitness = {self.fitness}'
