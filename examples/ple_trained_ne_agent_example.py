from python_ne.core.model_adapters.default_model_adapter import DefaultModelAdapter
from python_ne.core.model_adapters.keras_model_adapter import KerasModelAdapter
from python_ne.extra.ne_agent import NeAgent
from python_ne.extra.env_adapters.ple_env_adapter import PleEnvAdapter
from ple.games.flappybird import FlappyBird
from ple import PLE

if __name__ == '__main__':
    env = PLE(FlappyBird(), display_screen=True, force_fps=False)
    env.init()

    env_adapter = PleEnvAdapter(env=env)

    agent = NeAgent(
        env_adapter=env_adapter,
        model_adapter=KerasModelAdapter,
    )

    agent.load('ne_agent.json')
    while 1:
        agent.play()
