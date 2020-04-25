import gym
from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeAgent

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')

    env_adapter = GymEnvAdapter(env=env, render=False, continuous=True)

    agent = NeAgent(
        env_adapter=env_adapter,
        ne_type=NeNeuralNetwork,
        model_adapter=DefaultModelAdapter,
    )

    nn_config = (
        (env_adapter.get_input_shape(), 16, 'tanh'),
        (16, 'tanh'),
        (env_adapter.get_continuous_space_len(), 'tanh')
    )

    agent.train(
        number_of_generations=1000,
        population_size=200,
        selection_percentage=0.5,
        mutation_chance=0.1,
        fitness_threshold=20000,
        neural_network_config=nn_config,
    )

    agent.save('ne_agent.json')
