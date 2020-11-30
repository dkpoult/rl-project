import torch
import gym
from gym import spaces
import numpy as np
from collections import deque

def frame_stack_to_tensors(frames, use_glyphs = False, combine = True, flatten = True):
    """
    Used to convert a set of observation frames into suitably formatted tensors.
    Not actually used anymore because we don't stack frames.
    """
    tensors = []

    for i, frame in enumerate(frames):
        if use_glyphs:
            world = torch.IntTensor(frame["glyphs"])
        else:
            chars = torch.IntTensor(frame["chars"])
            colors = torch.IntTensor(frame["colors"])
            world = torch.stack((chars, colors), dim = 0)

        tensors.append(world)
        stats = torch.IntTensor(frame["blstats"])
        message = torch.IntTensor(frame["message"])
        inventory = torch.IntTensor(frame["inv_glyphs"])

        if combine:
            state = torch.cat((torch.flatten(world), stats, message, inventory))
            tensors.append(state)
        else:
            tensors.append([world, stats, message, inventory])

    if combine:
        to_return = torch.stack(tensors)

        if flatten:
            return torch.flatten(to_return)
        else:
            return to_return
    else:
        to_return = []
        for i in range(len(tensors[0])):
            to_return.append(torch.cat([frame[i] for frame in tensors]))
        return to_return


class FrameStack():
    def __init__(self, k):
        """
        Stacks the last k frames, turns it into an array of
        dictionaries.
        """
        self.k = k
        self.frames = deque([], maxlen = k)

    def reset(self, observation):
        for _ in range(self.k):
            self.frames.append(observation)

    def append(self, observation):
        if len(self.frames) == 0:
            self.reset(observation)
        else:
            self.frames.append(observation)

    def __call__(self):
        assert len(self.frames) == self.k
        return list(self.frames)


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._next_idx = 0
        self.size = size

    def __len__(self):
        return len(self._storage)

    def append(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= self.size:
            self._next_idx = 0

        if len(self._storage) < self.size:
            self._storage.append(data)        
        else:
            self._storage[self._next_idx] = data

        self._next_idx += 1

    def _encode_samples(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]

            state, action, reward, next_state, done = data

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __call__(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size = batch_size)
        return self._encode_samples(indices)
