from python_ne.core.ga.ga_neural_network import GaNeuralNetwork
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.core.ga.genetic_algorithm import GeneticAlgorithm
import numpy as np


def calc_fitness(element):
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], ])
    Y = np.array([0, 1, 1, 0])

    fitness = 0
    for x, y in zip(X, Y):
        prediction = np.argmax(element.get_output(x))
        if prediction == y:
            fitness += 1
    return fitness


if __name__ == '__main__':
    genetic_algorithm = GeneticAlgorithm(
        population_size=200,
        selection_percentage=0.9,
        mutation_chance=0.01,
        fitness_threshold=4,
        model_adapter=DefaultModelAdapter,
        neural_network_config=[((2,), 8, 'tanh'), (2, 'tanh')],
        console_log=True
    )

    genetic_algorithm.run(
        number_of_generations=1000,
        calculate_fitness_callback=calc_fitness
    )
    genetic_algorithm.save_log_data('file.csv')
    best_element = genetic_algorithm.get_best_element()

    print(f'the fitness of the best element is {best_element.raw_fitness}')
