import random


class GaNeuralNetwork:

    def __init__(self, model_adapter, neural_network_config=None, create_model=True, output_size=None,
                 input_shape=None, ):
        self.model_adapter = model_adapter
        self.neural_network_config = neural_network_config
        self.input_shape = input_shape
        self.output_size = output_size
        self.fitness = 0
        self.dense_layers = list()
        self.model = self.create_model() if create_model else None
        self.was_evaluated = False

    def create_model(self):
        raise NotImplementedError()

    def get_output(self, obs):
        return self.model.predict(obs)

    def crossover(self, other):
        raise NotImplementedError()

    def mutate(self):
        for layer in self.model.get_layers():
            weights, bias = layer.get_weights()

            prev_layer_neuron_count, current_layer_neuron_count = weights.shape

            for prev_neuron in range(prev_layer_neuron_count):
                for cur_neuron in range(current_layer_neuron_count):
                    if random.random() < 0.1:
                        weights[prev_neuron][cur_neuron] = random.uniform(-1, 1)

            for i in range(len(bias)):
                if random.random() < 0.1:
                    bias[i] = random.uniform(-1, 1)

            layer.set_weights((weights, bias))

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        # load is an static method in the adapter class
        self.model = self.model_adapter.load(file_path)

    def __str__(self):
        return f'fitness = {self.fitness}'
