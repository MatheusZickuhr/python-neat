import gym
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeAgent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    env_adapter = GymEnvAdapter(env=env, render=True)

    agent = NeAgent(
        env_adapter=env_adapter,
        population_size=200,
        input_shape=(8,),
        selection_percentage=0.5,
        mutation_chance=0.3,
        fitness_threshold=200
    )

    agent.load('ne_agent.json')
    agent.play()
