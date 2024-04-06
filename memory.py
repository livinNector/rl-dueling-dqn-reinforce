import torch
import numpy as np
from collections import namedtuple, deque
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


class ReplayMemory(object):
    Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

    def __init__(self, buffer_size, batch_size, seed=0):
        self.memory = deque([], maxlen=buffer_size)
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def push(self, transition_tuple):
        """Save a transition. If teminated next_state will be none"""
        # storing None as next state for terminated state this will be ignored while computing the target
        state, action, reward, next_state, terminated, truncated, _ = transition_tuple
        if terminated | truncated:
            next_state = None
        self.memory.append(self.Transition(state, action, reward, next_state))

    def sample(self):
        experiences = self.rng.sample(self.memory, k=self.batch_size)

        # batch of experiences to experience-batches
        experiences = self.Transition(*zip(*experiences))

        states = torch.from_numpy(np.array(experiences.state)).float().to(device)

        # only non terminal next_states
        is_not_none = lambda s: s is not None
        not_done = (
            torch.from_numpy(np.array(tuple(map(is_not_none, experiences.next_state))))
            .bool()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack(tuple(filter(is_not_none, experiences.next_state)))
            )
            .float()
            .to(device)
        )

        # discrete actions
        actions = torch.from_numpy(np.vstack(experiences.action)).long().to(device)
        rewards = torch.from_numpy(np.vstack(experiences.reward)).float().to(device)

        return states, actions, rewards, next_states, not_done

    def __len__(self):
        return len(self.memory)
