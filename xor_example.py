from python_neat.core.ga.genetic_algorithm import GeneticAlgorithm
import numpy as np


def calc_fitness(element):
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], ])
    Y = np.array([0, 1, 1, 0])

    fitness = 0
    for x, y in zip(X, Y):
        prediction = element.get_output(x.reshape(1, 2))
        if prediction == y:
            fitness += 1
    return fitness


genetic_algorithm = GeneticAlgorithm(
    population_size=200,
    input_shape=(2,),
    output_size=1,
    selection_percentage=0.5,
    mutation_chance=0.1,
    fitness_threshold=4
)

genetic_algorithm.run(
    number_of_generations=500,
    calculate_fitness_callback=calc_fitness
)

best_element = genetic_algorithm.get_best_element()

for e in genetic_algorithm.population:
    print(e)
print(f'the fitness of the best element is {best_element.fitness}')
