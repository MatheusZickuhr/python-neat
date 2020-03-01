from neat_agent import NeatAgent
import gym
from gym_env_adapter import GymEnvAdapter

env = gym.make('LunarLander-v2')

env_adapter = GymEnvAdapter(env=env)

agent = NeatAgent(env=env_adapter, population_size=500, input_shape=(8,), reward_log_path='rf1_neat_reward_log.txt')
agent.fit(generations=200, save_as='models/nrf1_neat.model')

