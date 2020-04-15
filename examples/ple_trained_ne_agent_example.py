from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.extra.ne_agent import NeAgent
from python_ne.extra.env_adapters.ple_env_adapter import PleEnvAdapter
from ple.games.pong import Pong
from ple import PLE

if __name__ == '__main__':
    env = PLE(Pong(), display_screen=True, force_fps=False)
    env.init()

    env_adapter = PleEnvAdapter(env=env)

    agent = NeAgent(
        env_adapter=env_adapter,
        ne_type=NeNeuralNetwork,
        model_adapter=DefaultModelAdapter,
    )

    agent.load('ne_agent.json')
    agent.play()
