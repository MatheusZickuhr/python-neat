from python_ne.core.ga_neural_network.neat_neural_network import NeatNeuralNetwork
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.core.ga.genetic_algorithm import GeneticAlgorithm
import numpy as np

from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork


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
        input_shape=(2,),
        output_size=2,
        selection_percentage=0.5,
        mutation_chance=0.1,
        fitness_threshold=4,
        ne_type=NeNeuralNetwork,
        model_adapter=DefaultModelAdapter,  # default or keras,
        neural_network_config=[128, 128],  # two hidden layers with 128 neurons each
        console_log=True
    )

    genetic_algorithm.run(
        number_of_generations=1000,
        calculate_fitness_callback=calc_fitness
    )
    genetic_algorithm.save_log_data('file.csv')
    best_element = genetic_algorithm.get_best_element()

    print(f'the fitness of the best element is {best_element.fitness}')
