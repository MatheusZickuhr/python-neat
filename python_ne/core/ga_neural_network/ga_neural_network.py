class GaNeuralNetwork:

    def __init__(self, backend_adapter, neural_network_config=None, create_model=True, output_size=None,
                 input_shape=None, ):
        self.backend_adapter = backend_adapter
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
        raise NotImplementedError()

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        # load is an static method in the adapter class
        self.model = self.backend_adapter.load(file_path)

    def __str__(self):
        return f'fitness = {self.fitness}'
