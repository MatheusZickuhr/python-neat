import gym
from python_neat.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_neat.extra.neat_agent import NeatAgent

env = gym.make('LunarLander-v2')

env_adapter = GymEnvAdapter(env=env, render=True)

agent = NeatAgent(
    env_adapter=env_adapter,
    population_size=100,
    input_shape=(8,),
    selection_percentage=0.2,
    mutation_chance=0.1
)

agent.train(number_of_generations=3, )

agent.save('test_model.model')
