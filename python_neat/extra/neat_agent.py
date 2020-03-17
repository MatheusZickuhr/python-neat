import keras

from python_neat.core.genetic_algorithm import GeneticAlgorithm


class NeatAgent:

    def __init__(self, env_adapter):
        self.env_adapter = env_adapter

    def train(self, number_of_generations, selection_percentage, mutation_chance, input_shape, population_size):

        genetic_algorithm = GeneticAlgorithm(
            population_size=population_size,
            input_shape=input_shape,
            output_size=self.env_adapter.get_n_actions(),
            selection_percentage=selection_percentage,
            mutation_chance=mutation_chance
        )
        genetic_algorithm.run(
            number_of_generations=number_of_generations,
            calculate_fitness_callback=self.calculate_fitness
        )

    def calculate_fitness(self, element):
        self.env_adapter.reset()
        done = False
        observation, _, _ = self.env_adapter.step(self.env_adapter.get_random_action())
        fitness = 0
        while not done:
            observation = keras.utils.normalize(observation)
            action = element.get_output(observation)
            observation, reward, done = self.env_adapter.step(action)
            fitness += reward if reward > 0 else 0
        return fitness

