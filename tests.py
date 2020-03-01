import unittest


class EnvAdaptersTest(unittest.TestCase):

    def test_gym_env(self):
        from gym_env_adapter import GymEnvAdapter
        import gym

        env = gym.make('LunarLander-v2')

        adapter = GymEnvAdapter(env=env)

        self.execute_env_adapter(adapter)

    def test_ple_adapter(self):
        from ple_env_adapter import PleEnvAdapter
        from ple.games.pong import Pong
        from ple import PLE

        env = PLE(Pong(), fps=30, display_screen=True, force_fps=False)
        env.init()

        adapter = PleEnvAdapter(env=env)

        self.execute_env_adapter(adapter)

    def execute_env_adapter(self, adapter):
        self.assertEqual(type(adapter.get_n_actions()), int)

        adapter.reset()

        random_action = adapter.get_random_action()

        self.assertIsNotNone(random_action)

        result = adapter.step(random_action)
        observation, reward, done = result

        self.assertEqual(len(result), 3)
        self.assertIsNotNone(observation)
        self.assertEqual(type(reward), float)
        self.assertEqual(type(done), bool)


if __name__ == '__main__':
    unittest.main()
