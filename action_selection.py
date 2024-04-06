import numpy as np
from numba import njit


class DecayingParam:
    """Class to be used of decaying parameters such as epsilon and tau"""

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.value = start

    def reset(self):
        self.value = self.start

    def update(self):
        self.value = max(self.end, self.value * self.decay)
        return self.value

    def __str__(self):
        return f"DecayingParam(start={self.start}, end={self.end}, decay={self.decay})"


# a helper function to sample according to distribution
@njit(fastmath=True)
def random_choice(prob):
    n_actions = prob.shape[0]
    prob = prob / prob.sum()
    r = np.random.random()
    p = 0.0
    for i in range(n_actions):
        p += prob[i]
        if p > r:
            return i
    return i


class ActionSelectionPolicy:
    def choose_action(self, Q, state):
        raise NotImplementedError


class NbEpsilonGreedy(ActionSelectionPolicy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, Q):
        return self._choose_action(self.epsilon, Q)

    @staticmethod
    @njit(fastmath=True)
    def _choose_action(epsilon, Q):
        if np.random.random() < 1 - epsilon:
            return Q.argmax()
        return random_choice(np.ones_like(Q) / Q.shape[0])

    def __str__(self):
        return f"e-greedy(epsilon={self.epsilon})"


@njit(fastmath=True)
def nb_softmax(p):
    p -= p.max()
    return np.exp(p) / np.exp(p.sum())


class NbSoftmax(ActionSelectionPolicy):
    def __init__(self, tau):
        self.tau = tau

    def choose_action(self, Q):
        return self._choose_action(self.tau, Q)

    @staticmethod
    @njit(fastmath=True)
    def _choose_action(tau, Q):
        return random_choice(nb_softmax(Q / tau))

    def __str__(self):
        return f"sofmax(tau={self.tau})"


class DecayingEpsilonGreedy(NbEpsilonGreedy):
    def __init__(self, epsilon: DecayingParam):
        self._epsilon = epsilon
        self.update = self._epsilon.update
        self.reset = self._epsilon.reset
        self.reset()

    @property
    def epsilon(self):
        return self._epsilon.value

    def __str__(self):
        return f"e-greedy(epsilon={self._epsilon})"


class DecayingSoftmax(NbSoftmax):
    def __init__(self, tau: DecayingParam):
        self._tau = tau
        self.update = self._tau.update
        self.reset = self._tau.reset
        self.reset()

    @property
    def tau(self):
        return self._tau.value

    def __str__(self):
        return f"softmax(tau={self._tau})"


# import numpy as np
# import matplotlib.pyplot as plt

# seed = 42
# np.random.seed(seed)

# # Sample Q
# Q = np.arange(1,5).reshape(-1)
# n = DecayingParam(1,.01,.995)
# e_greedy_policy = DecayingEpsilonGreedy(n)
# # e_greedy_policy.choose_action(Q,0)
# actions = []
# epsilons = []
# for i in range(1000):
#   actions.append(e_greedy_policy.choose_action(Q))
#   epsilons.append(e_greedy_policy.epsilon)
#   e_greedy_policy.update()

# plt.subplot(121)
# plt.hist(actions)
# plt.subplot(122)
# plt.plot(epsilons)
# plt.title(str(e_greedy_policy))
# plt.show()

# n = DecayingParam(4,.01,.995)

# softmax_policy = DecayingSoftmax(n)
# # softmax_policy.choose_action(Q,0)
# actions = []
# taus = []
# for i in range(200):
#   actions.append(softmax_policy.choose_action(Q))
#   taus.append(softmax_policy.tau)
#   softmax_policy.update()

# plt.subplot(121)
# plt.hist(actions)
# plt.subplot(122)
# plt.plot(taus)
# plt.title(str(softmax_policy))
# plt.show()
