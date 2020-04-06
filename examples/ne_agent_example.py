import gym
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeAgent

env = gym.make('LunarLander-v2')

env_adapter = GymEnvAdapter(env=env, render=True)

agent = NeAgent(
    env_adapter=env_adapter,
    population_size=100,
    input_shape=(8,),
    selection_percentage=0.2,
    mutation_chance=0.1,
    fitness_threshold=200
)

agent.train(number_of_generations=3, )
