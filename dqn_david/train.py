import numpy as np
import math
import gym
import nle
import random
import torch
from os import path

from shared.utils import FrameStack, ReplayBuffer
from dqn_david.MyAgent import MyAgent

def observation_to_input(observation):
    """
    Takes an observation and transforms it into the format
    to be stored in the buffer and/or provided to our model.
    """

    # Just takes the basic data, doesn't do anything
    # fancy for now.
    glyphs = np.array(observation["glyphs"])
    stats = np.array(observation["blstats"])

    return [glyphs, stats]


def train():
    # The general training hyper params, not just for the model
    hyper_params = {
        "seed": None,  # which seed to use
        "replay-buffer-size": int(1e5),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for the optimizer
        "discount-factor": 0.999,  # discount factor
        "num-steps": int(2e1),  # total number of steps to run the environment for
        "batch-size": 64,  # number of transitions to optimize at the same time
        "learning-starts": 1e5,  # number of steps before learning starts
        "learning-freq": 1e2,  # number of iterations between every optimization step
        "target-update-freq": 5e3,  # number of iterations between every target network update
        "eps-start": 1.00,  # e-greedy start threshold
        "eps-end": 1e-3,  # e-greedy end threshold
        "eps-fraction": 5e-2,  # fraction of num-steps
        "print-freq": 10, # number of episodes between detail logs
        "save-freq": 100000, # number of steps between model saves
    }

    # Use the score environment
    env = gym.make("NetHackScore-v0")
    
    # If we have a seed then use it for everything we can
    if hyper_params["seed"] is not None:
        np.random.seed(hyper_params["seed"])
        random.seed(hyper_params["seed"])
        env.seed(hyper_params["seed"])

    # Set up our replay buffer
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    # Initialize our agent from our hyper parameters, with our replay buffer too
    agent = MyAgent(env.observation_space, env.action_space, replay_buffer = replay_buffer, \
        learning_rate = hyper_params["learning-rate"], batch_size = hyper_params["batch-size"], \
        discount = hyper_params["discount-factor"], train = True)

    # The point where our epsilon stops decreasing
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])

    # Stats tracking arrays
    episode_rewards = [0.0]
    episode_lengths = [0]

    # Kick off the learning process!
    state = env.reset()
    
    # Go through the maximum number of steps
    for t in range(1, hyper_params["num-steps"]):
        # Find out the epsilon at this point
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )

        # Sample for random action or agent action
        sample = random.random()
        
        if sample < eps_threshold:
            # Just sample from the action space
            action = env.action_space.sample()
        else:
            # Get it from our agent
            action = agent.act(observation_to_input(state))

        # Do the action, duh
        new_state, reward, done, _ = env.step(action)

        # Add the augmented state to our replay buffer
        replay_buffer.append(observation_to_input(state), action, \
                reward, observation_to_input(new_state), done)

        # Update our state
        state = new_state
        
        # Update our stats
        episode_rewards[-1] += reward
        episode_lengths[-1] += 1

        if done:
            # Reset the env
            state = env.reset()

            # Log the nice details
            print("\tFinished episode: {}".format(len(episode_rewards)))
            print("\t\tGot reward:     {}".format(episode_rewards[-1]))
            print("\t\tDuration:       {}".format(episode_lengths[-1]))

            # Save our result graphs so far
            np.save("/home/david/university/RL/rl-project/rewards", episode_rewards)
            np.save("/home/david/university/RL/rl-project/lengths", episode_lengths)

            # Start a few episode tracker
            episode_rewards.append(0.0)
            episode_lengths.append(0)

        if (t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0):
            # Time to optimise our agent
            loss = agent.optimise_td_loss()

        if (t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0):
            # Time to update our target network
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if t % hyper_params["save-freq"] == 0:
            # Save our model
            curr_path = path.abspath(path.dirname(__file__))
            torch.save(agent.pred_model.state_dict(), path.join(curr_path, "models/tmp_dqn_trained.pt"))

        if (done and hyper_params["print-freq"] is not None 
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            # Print out our running details
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

    # Save the model when it's done
    curr_path = path.abspath(path.dirname(__file__))
    torch.save(agent.pred_model.state_dict(), path.join(curr_path, "models/dqn_trained.pt"))

train()
