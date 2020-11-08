from dqn_david.DQN import DQN
from shared.AbstractAgent import AbstractAgent
from shared.utils import FrameStack, frame_stack_to_tensors
import torch
from torch import nn, FloatTensor, LongTensor
import os

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, learning_rate = 1e-3,
            discount = 1.0, batch_size = 32, replay_buffer = None, train = False, from_file = False):
        
        print("Creating agent in", "train mode" if train else "play mode")
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.discount = discount

        self.frame_stack = FrameStack(4)
        self.replay_buffer = replay_buffer

        self.pred_model = DQN(out_size = action_space.n)
        self.target_model = DQN(out_size = action_space.n)

        if from_file:
            curr_path = path.abspath(path.dirname(__file__))
            if os.path.exists(os.path.join(curr_path, "models/dqn_trained.pt")):
                self.pred_model.load_state_dict(torch.load(os.path.join(curr_path, "models/dqn_trained.pt")))
                self.update_target_network()
            else:
                print("No file found to load from, starting training from scratch.")

        self.optimiser = torch.optim.RMSprop(self.pred_model.parameters(), lr = learning_rate)
        self.loss = nn.SmoothL1Loss()

        print("Utilizing", device)

    def act(self, observation):
        self.frame_stack.append(observation)

        x = frame_stack_to_tensors(self.frame_stack(), combine = False)
        
        values = self.pred_model([x])

        return values.argmax().item()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        batch = self.replay_buffer(self.batch_size)
        (states, actions, rewards, next_states, dones) = batch
        
        states = [frame_stack_to_tensors(state, combine = False) for state in states]
        next_states = [frame_stack_to_tensors(state, combine = False) for state in next_states]

        actions = LongTensor(actions).to(device)
        rewards = FloatTensor(rewards).to(device)
        dones = FloatTensor(dones).to(device)

        # Get the values of the current states' actions
        current_values = self.pred_model(states)
        # Get the state-action values that we actually took
        current_values = current_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get the maximum state-action values of the next states
        next_values = self.target_model(next_states).max(axis = 1)[0]
        
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
