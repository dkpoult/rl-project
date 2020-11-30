import gym
import nle

def test():
    """
    Barebones implementation to test that the environment is actually
    functioning on our machines.
    """
    env = gym.make("NetHackScore-v0")

    state = env.reset()
    done = False
    step = 0

    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)

        step += 1

test()
