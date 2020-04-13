import numpy as np
from python_ne.core.ga.genetic_algorithm import GeneticAlgorithm
from python_ne.core.neural_network import normalizer


class NeAgent:

    def __init__(self, env_adapter, model_adapter, ne_type):
        self.env_adapter = env_adapter
        self.model_adapter = model_adapter
        self.ne_type = ne_type
        self.best_element = None
        self.genetic_algorithm = None

    def train(self, number_of_generations, selection_percentage, mutation_chance,
              input_shape, population_size, fitness_threshold, neural_network_config):
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=population_size,
            input_shape=input_shape,
            output_size=self.env_adapter.get_n_actions(),
            selection_percentage=selection_percentage,
            mutation_chance=mutation_chance,
            fitness_threshold=fitness_threshold,
            ne_type=self.ne_type,
            model_adapter=self.model_adapter,
            neural_network_config=neural_network_config
        )

        self.genetic_algorithm.run(
            number_of_generations=number_of_generations,
            calculate_fitness_callback=self.calculate_fitness
        )

        self.best_element = self.genetic_algorithm.get_best_element()

    def calculate_fitness(self, element):
        return self.play(element)

    def save(self, file_path):
        self.best_element.save(file_path)

    def load(self, file_path):
        self.best_element = self.ne_type(
            create_model=False,
            model_adapter=self.model_adapter
        )
        self.best_element.load(file_path)

    def play(self, element=None):
        element = self.best_element if element is None else element
        self.env_adapter.reset()
        done = False
        observation, _, _ = self.env_adapter.step(self.env_adapter.get_random_action())
        fitness = 0
        while not done:
            observation = normalizer.normalize(observation)
            action = np.argmax(element.get_output(np.array(observation)))
            observation, reward, done = self.env_adapter.step(action)
            fitness += reward if reward > 0 else 0
        return fitness
