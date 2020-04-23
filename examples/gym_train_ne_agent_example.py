import gym
from python_ne.core.ga_neural_network.ne_neural_network import NeNeuralNetwork
from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeAgent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    env_adapter = GymEnvAdapter(env=env, render=False)

    agent = NeAgent(
        env_adapter=env_adapter,
        ne_type=NeNeuralNetwork,
        model_adapter=DefaultModelAdapter,
    )

    agent.train(
        number_of_generations=1000,
        population_size=500,
        selection_percentage=0.5,
        mutation_chance=0.1,
        fitness_threshold=20000,
        neural_network_config=(
            (512, 'sigmoid'),
            (512, 'sigmoid'),
            (env_adapter.get_n_actions(), 'sigmoid')
        ),
        play_n_times=5,
        max_n_steps=250,
        reward_if_max_step_reached=-100
    )

    agent.save('ne_agent.json')
