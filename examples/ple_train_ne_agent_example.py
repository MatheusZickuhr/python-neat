import os

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork
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
        ne_type=NeNeuralNetwork,
        model_adapter=DefaultModelAdapter,
    )

    agent.train(
        number_of_generations=150,
        population_size=500,
        selection_percentage=0.5,
        mutation_chance=0.3,
        fitness_threshold=45,
        neural_network_config=[
            (env_adapter.get_input_shape(), 512, 'sigmoid'),
            (512, 'sigmoid'),
            (env_adapter.get_n_actions(), 'sigmoid')
        ]
    )

    agent.save('ne_agent.json')
