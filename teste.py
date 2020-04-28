import gym
import resource


def main():
    env = gym.make('LunarLander-v2')
    for i in range(3_000_000):
        env.reset()
        ram_peak_python_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // int(1e6)
        if i % 1000 == 0:
            print(ram_peak_python_mb)


if __name__ == "__main__":
    main()
