
# python-ne
A neuroevolution library for python

# how to install

```
git clone https://github.com/MatheusZickuhr/python-ne.git
cd python-ne
pip install -r requirements.txt
```

# how to install (pip)

```
pip install python-ne
```

# example (xor)

```python
from python_ne.core.ga.csv_logger import CsvLogger
from python_ne.core.ga.console_logger import ConsoleLogger
from python_ne.core.ga.matplotlib_logger import MatplotlibLogger
from python_ne.core.ga.mutation_strategies import Mutation1
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.core.ga.genetic_algorithm import GeneticAlgorithm
from python_ne.core.ga.crossover_strategies import Crossover1
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
        crossover_strategy=Crossover1(),
        mutation_strategy=Mutation1(),
        neural_network_config=[((2,), 8, 'tanh'), (2, 'tanh')],
    )

    csv_logger = CsvLogger()
    genetic_algorithm.add_observer(csv_logger)
    matplotlib_logger = MatplotlibLogger()
    genetic_algorithm.add_observer(matplotlib_logger)
    genetic_algorithm.add_observer(ConsoleLogger())

    genetic_algorithm.run(
        number_of_generations=1000,
        calculate_fitness_callback=calc_fitness
    )

    csv_logger.save('file.csv')
    matplotlib_logger.save_fitness_chart('fitness.png')
    matplotlib_logger.save_time_chart('time.png')
    matplotlib_logger.save_std_chart('std.png')

    best_element = genetic_algorithm.get_best_element()

    print(f'the fitness of the best element is {best_element.raw_fitness}')

```
