from dqn_david.DQN import DQN
from shared.AbstractAgent import AbstractAgent
import torch
from torch import nn, FloatTensor, LongTensor
import os
from os import path
import numpy as np

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

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

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, learning_rate = 1e-3,
            discount = 1.0, batch_size = 32, replay_buffer = None, train = False, from_file = True):
        
        print("Creating agent in", "train mode" if train else "play mode")
        
        # Store the basic information
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.discount = discount
        self.replay_buffer = replay_buffer

        # Set up our policy and target networks
        self.pred_model = DQN(out_size = action_space.n)
        self.target_model = DQN(out_size = action_space.n)

        # Load the weights from a file if they exists
        if from_file:
            print("Trying to load from file.")
            curr_path = path.abspath(path.dirname(__file__))
            if os.path.exists(os.path.join(curr_path, "models/dqn_trained.pt")):
                print("Found a file to load from!")
                self.pred_model.load_state_dict(torch.load(os.path.join(curr_path, "models/dqn_trained.pt")))
                self.update_target_network()
            else:
                print("No file found to load from, starting training from scratch.")

        # Set up our optimiser and loss function
        self.optimiser = torch.optim.RMSprop(self.pred_model.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()

        print("Utilizing", device)

    def act(self, observation):
        """
        Only intended for single observations, gives back an action index.
        """
        state = observation_to_input(observation)

        # Gets the glyph and stats tensors suitably formatted
        x_glyphs = torch.FloatTensor([state[0]]).unsqueeze(0).to(device)
        x_stats = torch.FloatTensor([state[1]]).unsqueeze(0).to(device)

        # Get the action values
        values = self.pred_model([x_glyphs, x_stats])

        # Give back the best
        return values.argmax().item()

    def optimise_td_loss(self):
        """
        Optimise on the TD-error over a minibatch of our stored experience.
        """
        # Get a batch of experiences
        batch = self.replay_buffer(self.batch_size)
        (states, actions, rewards, next_states, dones) = batch

        # Get the features from our "current" states
        x_glyphs = torch.stack([torch.from_numpy(state[0]).unsqueeze(0) for state in states]).float().to(device)
        x_stats = torch.stack([torch.from_numpy(state[1]).unsqueeze(0) for state in states]).float().to(device)

        # Get the features from our "next" states
        next_x_glyphs = torch.stack([torch.from_numpy(state[0]).unsqueeze(0) for state in next_states]).float().to(device)
        next_x_stats = torch.stack([torch.from_numpy(state[1]).unsqueeze(0) for state in next_states]).float().to(device)

        # Get all our other TD information to tensor form
        actions = LongTensor(actions).to(device)
        rewards = torch.tanh(FloatTensor(rewards).to(device))
        dones = FloatTensor(dones).to(device)

        # Get the values of the "current" states' actions
        current_values = self.pred_model([x_glyphs, x_stats])
        # Get the state-action values that we actually took
        current_values = current_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Get the maximum state-action values of the next states
            next_values = self.target_model([next_x_glyphs, next_x_stats]).max(axis = 1)[0]
        
        # Calculate the targets
        targets = rewards + (1 - dones) * self.discount * next_values
        
        # Find the losses
        losses = self.loss(current_values, targets)

        # Perform back-prop
        self.optimiser.zero_grad()
        losses.backward()
        self.optimiser.step()

        return losses

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_model.load_state_dict(self.pred_model.state_dict())
