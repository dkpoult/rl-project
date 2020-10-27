import numpy as np
import gym
import nle
import random
import torch
from shared.utils import FrameStack, ReplayBuffer
from dqn_david.MyAgent import MyAgent

def train():
    env = gym.make("NetHackScore-v0")

    state = env.reset()
    done = False

    while not done:
        state, reward, done, _ = env.step(env.action_space.sample())

        print(reward)


if __name__ == '__main__':
    # Initialise environment
    train()
