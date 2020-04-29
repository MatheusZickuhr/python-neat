import os

from python_ne.core.ga.console_logger import ConsoleLogger
from python_ne.core.ga.crossover_strategies import Crossover1, NoCrossover
from python_ne.core.ga.mutation_strategies import Mutation1
from python_ne.core.model_adapters.keras_model_adapter import KerasModelAdapter

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"


from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.extra.ne_agent import NeAgent
from python_ne.extra.env_adapters.ple_env_adapter import PleEnvAdapter
from ple.games.flappybird import FlappyBird
from ple import PLE

if __name__ == '__main__':
    env = PLE(FlappyBird(), display_screen=False, force_fps=True)
    env.init()

    env_adapter = PleEnvAdapter(env=env)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=KerasModelAdapter,
    )

    agent.train(
        number_of_generations=150,
        population_size=500,
        selection_percentage=0.9,
        mutation_chance=0.01,
        fitness_threshold=50,
        crossover_strategy=NoCrossover(),
        mutation_strategy=Mutation1(),
        neural_network_config=[
            (env_adapter.get_input_shape(), 16, 'tanh'),
            (env_adapter.get_n_actions(), 'tanh')
        ],
        loggers=(ConsoleLogger(),)
    )

    agent.save('ne_agent.json')
