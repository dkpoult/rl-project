from dqn_david.DQN import DQN
from shared.AbstractAgent import AbstractAgent
from shared.utils import FrameStack, frame_stack_to_tensors
import torch
from torch import nn, FloatTensor, LongTensor

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

        self.pred_model = DQN(in_size = 14616, out_size = action_space.n)
        self.target_model = DQN(in_size = 14616, out_size = action_space.n)

        if from_file:
            self.pred_model.load_state_dict(torch.load("/home/david/university/RL/Project/models/dqn_trained.pt"))
            self.update_target_network()

        self.optimiser = torch.optim.Adam(self.pred_model.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()

        print("Utilizing", device)

    def act(self, observation):
        self.frame_stack.append(observation)

        x = frame_stack_to_tensors(self.frame_stack(), combine = True, flatten = True)
        
        values = self.pred_model(x)

        return values.argmax().item()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        batch = self.replay_buffer(self.batch_size)
        (states, actions, rewards, next_states, dones) = batch
        
        state_stacks = torch.stack([frame_stack_to_tensors(state) for state in states])
        next_state_stacks = torch.stack([frame_stack_to_tensors(state) for state in next_states])

        states = state_stacks.float().to(device)
        next_states = next_state_stacks.float().to(device)
        
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
