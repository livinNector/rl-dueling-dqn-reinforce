import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from train_utils import train
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(1),
        )

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.network(x)


class ReinforceAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma,
        hidden_size=128,
        learning_rate=1e-4,
        baseline=False,
        baseline_lr=1e-2,
        seed=0,
    ):
        self.policy_network = PolicyNetwork(state_size, hidden_size, action_size).to(
            device
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs = []  # To keep track of the log of action probs
        self.policy_network.train()
        self.baseline = baseline
        if self.baseline:
            # linear function approximator
            self.value = nn.Linear(state_size, 1).to(device)
            # for gradient decent
            self.value_optimizer = optim.SGD(self.value.parameters(), lr=baseline_lr)
            self.states = []
        torch.manual_seed(seed)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def returns_from_rewards(self, rewards):
        returns = deque()
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        return returns

    def step(self, episode_accumulate):
        """called after each episode"""
        returns = self.returns_from_rewards(episode_accumulate["reward"])
        returns = torch.tensor(returns).float().to(device)

        if self.baseline:
            states = (
                torch.from_numpy(np.array(episode_accumulate["state"]))
                .float()
                .to(device)
            )
            values = self.value(states).squeeze()
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            returns = returns - values.detach()

        log_probs = torch.cat(self.log_probs).to(device)
        loss = (-log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []

    def train(
        self, env, n_episodes=1000, max_t=1000, reward_window=100, reward_threshold=None,verbose=False,plot=False
    ):
        """Returns the rewards"""
        return train(
            env,
            self,
            store_fields=["state", "reward"],
            episode_callbacks=[self.step],
            n_episodes=n_episodes,
            max_t=max_t,
            reward_window=reward_window,
            reward_threshold=reward_threshold,
            verbose=verbose,
            plot=plot
        )
