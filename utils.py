import numpy as np
import gymnasium as gym
from collections import namedtuple


TransitionTuple = namedtuple(
    "TransitionTuple",
    field_names=[
        "state",
        "action",
        "reward",
        "next_state",
        "terminated",
        "truncated",
        "info",
    ],
)


# @title Run episode function with callback
def run_episode(env: gym.Env, get_action=None, step_callbacks=None, max_t=1000):
    """
    Runs through one episode with the environment. time step is added to the info
    Args:
    env - gymnaisum environment
    step_callbacks - list of functions that accepts an TransitionTuple
                    that is called after every step
    """

    time_step = 0
    state, info = env.reset()
    terminated = truncated = False
    if get_action is None:
        get_action = lambda state: env.action_space.sample()

    if step_callbacks is None:
        step_callbacks = []

    for time_step in range(env.spec.max_episode_steps):
        action = get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        info["time_step"] = time_step
        if time_step == max_t - 1:
            truncated = 1

        step_tuple = TransitionTuple(
            state, action, reward, next_state, terminated, truncated, info
        )

        for step_callback in step_callbacks:
            step_callback(step_tuple)

        state = next_state

        if terminated | truncated:
            break


class EpisodeAccumulateCallback:
    def __init__(self, store_fields=None):
        self.store_fields = store_fields
        self.reset()

    def reset(self):
        self.reward = 0
        if self.store_fields:
            self.store = {value: [] for value in self.store_fields}

    def __call__(self, transition_tuple):
        self.reward += transition_tuple.reward
        if self.store_fields:
            for field in self.store_fields:
                self.store[field].append(transition_tuple.__getattribute__(field))


# @title generic train function
import matplotlib.pyplot as plt


def reward_plot(scores, message=""):
    plt.plot(scores)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("reward curve " + message)
    plt.show()


def train(
    env: gym.Env,
    agent,
    store_fields=None,
    episode_callbacks=None,
    step_callbacks=None,
    n_episodes=10000,
    max_t=1000,
    reward_window=100,
    reward_threshold=None,
    verbose=False,
    plot=False,
):
    # this is always called
    episode_accumulate = EpisodeAccumulateCallback(store_fields=store_fields)

    all_rewards = []
    if reward_threshold is None:
        reward_threshold = env.spec.reward_threshold

    if episode_callbacks is None:
        episode_callbacks = []

    if step_callbacks is None:
        step_callbacks = []

    for i_episode in range(1, n_episodes + 1):
        episode_accumulate.reset()
        run_episode(env, agent.act, [*step_callbacks, episode_accumulate], max_t=max_t)

        all_rewards.append(episode_accumulate.reward)
        average_reward = np.mean(all_rewards[-reward_window:])

        for episode_callback in episode_callbacks:
            episode_callback(episode_accumulate.store)
        if verbose:
            print(f"\rEpisode {i_episode}\tAverage reward: {average_reward:.2f}", end="")
            if i_episode % 100 == 0:
                print()  # retain the result
        if average_reward >= reward_threshold:
            print(
                f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {average_reward:.2f}"
            )
            break

    if plot:
        reward_plot(
            all_rewards, message=f"Env:{env.spec.name}, Agent:{agent.__class__.__name__}"
        )
    return all_rewards
