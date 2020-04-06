import random

from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork
from python_ne.core.ga.random_probability_selection import RandomProbabilitySelection
from tqdm import tqdm
from python_ne.core.ga_neural_network.neat_neural_network import NeatNeuralNetwork


def get_ga_neural_network_class_type(value):
    if value == 'ne':
        return NeNeuralNetwork
    elif value == 'neat':
        return NeatNeuralNetwork
    else:
        raise Exception(f'\'{value}\' ne_type not found, choices are ne or neat')


class GeneticAlgorithm:

    def __init__(self, population_size, selection_percentage, mutation_chance, input_shape, output_size,
                 fitness_threshold, ne_type):
        self.population_size = population_size
        self.input_shape = input_shape
        self.output_size = output_size
        self.ne_nn_class_type = get_ga_neural_network_class_type(ne_type)
        self.population = self.create_population()
        self.number_of_selected_elements = int(len(self.population) * selection_percentage)
        self.mutation_chance = mutation_chance
        self.fitness_threshold = fitness_threshold

    def create_population(self):
        return [self.ne_nn_class_type(input_shape=self.input_shape, output_size=self.output_size)
                for _ in tqdm(range(self.population_size), unit='population element created')]

    def run(self, number_of_generations, calculate_fitness_callback):
        for _ in tqdm(range(number_of_generations), unit='generation'):
            self.calculate_fitness(calculate_fitness_callback)
            new_elements = self.crossover()
            self.mutate(new_elements)
            self.recycle(new_elements)
            if self.get_best_element().fitness >= self.fitness_threshold:
                break

    def crossover(self):
        total_fitness = sum([element.fitness for element in self.population])
        random_probability_selection = RandomProbabilitySelection()
        random_probability_selection.add_elements(
            [{'data': element, 'probability': element.fitness / total_fitness} for element in self.population]
        )
        selected_elements = random_probability_selection.perform_selection(self.number_of_selected_elements)

        new_elements = []

        # iterate by pairs of elements
        for element1, element2 in zip(selected_elements[0::2], selected_elements[1::2]):
            new_elements.extend(element1.crossover(element2))

        return new_elements

    def mutate(self, new_elements):
        for element in new_elements:
            if random.random() <= self.mutation_chance:
                element.mutate()

    def recycle(self, new_elements):
        self.population.sort(key=lambda element: element.fitness)

        for index, new_element in enumerate(new_elements):
            self.population[index] = new_element

    def calculate_fitness(self, calculate_fitness_callback):
        for element in self.population:
            fitness = calculate_fitness_callback(element)
            element.fitness = fitness

    def get_best_element(self):
        self.population.sort(key=lambda element: element.fitness)
        return self.population[-1]
