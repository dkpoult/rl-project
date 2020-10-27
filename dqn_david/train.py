import numpy as np
import gym
import nle
import random
import torch
from shared.utils import FrameStack, ReplayBuffer
from dqn_david.MyAgent import MyAgent

def train():
    hyper_params = {
        "seed": 432,  # which seed to use
        "replay-buffer-size": int(1e4),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for the optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e9),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 1e4,  # number of steps before learning starts
        "learning-freq": 1e2,  # number of iterations between every optimization step
        "target-update-freq": 5e3,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.005,  # fraction of num-steps
        "print-freq": 10,
        "save-freq": 100000,
    }

    env = gym.make("NetHackScore-v0")

    if hyper_params["seed"] is not None:
        np.random.seed(hyper_params["seed"])
        random.seed(hyper_params["seed"])
        env.seed(hyper_params["seed"])

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    # TODONE Create dqn agent
    agent = MyAgent(env.observation_space, env.action_space, replay_buffer = replay_buffer, \
        learning_rate = hyper_params["learning-rate"], batch_size = hyper_params["batch-size"], \
        discount = hyper_params["discount-factor"], train = True, from_file = True)

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    episode_lengths = [0]
    frame_stack = FrameStack(4)

    state = env.reset()
    frame_stack.append(state)
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()
        
        if sample < eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        new_state, reward, done, _ = env.step(action)
        # env.render()

        state_stack = frame_stack()
        frame_stack.append(new_state)

        replay_buffer.append(state_stack, action, reward, frame_stack(), done)

        state = new_state
        episode_rewards[-1] += reward
        episode_lengths[-1] += 1
        if done:
            state = env.reset()

            print("\tFinished episode: {}".format(len(episode_rewards)))
            print("\t\tGot reward:     {}".format(episode_rewards[-1]))
            print("\t\tDuration:       {}".format(episode_lengths[-1]))

            episode_rewards.append(0.0)
            episode_lengths.append(0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            loss = agent.optimise_td_loss()

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()
            # print("Updating target network!")

        num_episodes = len(episode_rewards)

        if t % hyper_params["save-freq"] == 0:
            torch.save(agent.pred_model.state_dict(), "/home/david/university/RL/Project/models/tmp_dqn_trained.pt")

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

    torch.save(agent.pred_model.state_dict(), "/home/david/university/RL/Project/models/dqn_trained.pt")


if __name__ == '__main__':
    # Initialise environment
    train()
