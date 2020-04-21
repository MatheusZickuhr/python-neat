import gym

from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeAgent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    env_adapter = GymEnvAdapter(env=env, render=True)

    agent = NeAgent(
        env_adapter=env_adapter,
        ne_type=NeNeuralNetwork,
        model_adapter=DefaultModelAdapter,
    )

    agent.load('ne_agent.json')
    while 1:
        print(agent.play())
