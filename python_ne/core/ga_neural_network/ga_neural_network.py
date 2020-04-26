import random


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

    def crossover(self, other):
        raise NotImplementedError()

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

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        # load is an static method in the adapter class
        self.model = self.model_adapter.load(file_path)

    def __str__(self):
        return f'fitness = {self.fitness}'
