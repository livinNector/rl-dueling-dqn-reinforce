import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from memory import ReplayMemory
from utils import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


class FCNetwork(nn.Module):
    def __init__(self, input_size, fc_layer_sizes, output_size):
        super(FCNetwork, self).__init__()
        layer_sizes = [input_size, *fc_layer_sizes, output_size]
        self.network = nn.Sequential(
            *(
                layer
                for layer_input, layer_output in zip(layer_sizes, layer_sizes[1::])
                for layer in [nn.Linear(layer_input, layer_output), nn.ReLU()]
            )
        )

    def forward(self, state):
        return self.network(state)


class DuelingDQNetwork(nn.Module):
    def __init__(self, state_size, fc_layer_sizes, action_size, network_type="mean"):
        """
        Builds a Dueling DQN Network

        Args:
        network_type - 'mean' or 'max'

        """
        super().__init__()
        self.base_network = FCNetwork(
            state_size, fc_layer_sizes[:-1], fc_layer_sizes[-1]
        )
        self.value_network = nn.Linear(fc_layer_sizes[-1], 1)
        self.advantage_network = nn.Linear(fc_layer_sizes[-1], action_size)
        self.network_type = network_type

        if network_type == "mean":

            def forward(state):
                b = F.relu(self.base_network(state))
                V = self.value_network(b)
                A = self.advantage_network(b)
                return V + (A - A.mean())

        elif network_type == "max":

            def forward(state):
                b = F.relu(self.base_network(state))
                V = self.value_network(b)
                A = self.advantage_network(b)
                return V + (A - A.max())

        else:
            raise Exception("network type can be either 'mean' or 'max'")

        self.forward = forward


# @title  Implementation of DQN
class DuelingDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma,  # env params
        # hyper params
        network_layer_sizes,
        network_type,  # network params
        action_selector,
        buffer_size,
        batch_size,
        target_update_rate,
        learning_rate,  # memory and training params
        seed,
    ):
        """
        Dueling DQN Agent with
          - Experience Replay
          - Hard Target Updates
          - Gradient Clipping
          - Adam Optimizer
          - MSE LOSS
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.q_network = DuelingDQNetwork(
            state_size, network_layer_sizes, action_size, network_type
        ).to(device)
        self.target_q_network = DuelingDQNetwork(
            state_size, network_layer_sizes, action_size, network_type
        ).to(device)

        self.gamma = gamma
        self.action_selector = action_selector

        self.batch_size = batch_size
        self.target_update_rate = target_update_rate
        self.learning_rate = learning_rate
        self.memory = ReplayMemory(buffer_size, batch_size, seed)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.t_step = 0

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        action_values = action_values.cpu().float().data.numpy().squeeze()

        action = self.action_selector.choose_action(action_values)
        self.action_selector.update()

        return action

    def step(self, transition_tuple):
        # add experience to memory
        self.memory.push(transition_tuple)

        # sample from memory if enough samples are there
        if len(self.memory) >= self.batch_size:
            self.learn()

        self.t_step = (self.t_step + 1) % self.target_update_rate
        if self.t_step == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def learn(self):
        """+E EXPERIENCE REPLAY PRESENT"""
        states, actions, rewards, next_states, not_done = self.memory.sample()

        # max Q values from target network
        q_target_next = torch.zeros(self.batch_size, device=device)

        self.target_q_network.eval()
        with torch.no_grad():
            q_target_next[not_done] = self.target_q_network(next_states).max(-1).values
            # compute target for current state
            q_target = rewards + (self.gamma * q_target_next.unsqueeze(-1))
        self.target_q_network.train()
        
        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_value_(self.q_network.parameters(), 1)

        self.optimizer.step()

    def train(
        self,
        env,
        n_episodes: int = 10000,
        max_t: int = 1000,
        reward_window: int = 100,
        reward_threshold: int | None = None,
    ):
        self.action_selector.reset()
        return train(
            env,
            self,
            step_callbacks=[self.step],
            n_episodes=n_episodes,
            max_t=max_t,
            reward_window=reward_window,
            reward_threshold=reward_threshold,
        )
