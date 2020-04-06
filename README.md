# python-ne
A neuroevolution library for python


# how to install

```
git clone https://github.com/MatheusZickuhr/python-ne.git
cd python-ne
pip install -r requirements.txt
```

# example (xor)

```python
from python_ne.core.ga.genetic_algorithm import GeneticAlgorithm
import numpy as np


def calc_fitness(element):
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], ])
    Y = np.array([0, 1, 1, 0])

    fitness = 0
    for x, y in zip(X, Y):
        prediction = element.get_output(x)
        if prediction == y:
            fitness += 1
    return fitness


genetic_algorithm = GeneticAlgorithm(
    population_size=200,
    input_shape=(2,),
    output_size=2,
    selection_percentage=0.5,
    mutation_chance=0.1,
    fitness_threshold=4,
    ne_type='ne' # ne or neat
)

genetic_algorithm.run(
    number_of_generations=1000,
    calculate_fitness_callback=calc_fitness
)

best_element = genetic_algorithm.get_best_element()


print(f'the fitness of the best element is {best_element.fitness}')

```
