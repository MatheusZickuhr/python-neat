# python-neat
A neuroevolution of augmenting topologies library for python


# how to install

```
git clone https://github.com/MatheusZickuhr/python-neat.git
cd python-neat
pip install -r requirements.txt
```

# example

```
from neat_agent import NeatAgent
import gym


env = gym.make('LunarLander-v2')
agent = NeatAgent(env=env, population_size=500, input_shape=(8,), reward_log_path='rf1_neat_reward_log.txt')
agent.fit(generations=200, save_as='models/nrf1_neat.model')

```