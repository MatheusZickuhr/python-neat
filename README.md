# python-neat
A neuroevolution of augmenting topologies library for python


# how to install

```
git clone https://github.com/MatheusZickuhr/python-neat.git
cd python-neat
pip install -r requirements.txt
```

# example

```python
import gym
from python_ne.extra.env_adapters.gym_env_adapter import GymEnvAdapter
from python_ne.extra.ne_agent import NeatAgent

env = gym.make('LunarLander-v2')

env_adapter = GymEnvAdapter(env=env, render=True)

agent = NeatAgent(env_adapter=env_adapter, )

agent.train(
    population_size=100,
    input_shape=(8,),
    number_of_generations=3,
    selection_percentage=0.2,
    mutation_chance=0.1
)
```