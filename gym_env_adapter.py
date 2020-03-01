from env_adapter import EnvAdapter


class GymEnvAdapter(EnvAdapter):

    def reset(self):
        self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, float(reward), done

    def get_n_actions(self):
        return self.env.action_space.n

    def get_random_action(self):
        return self.env.action_space.sample()
