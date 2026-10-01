import numpy as np

from src.reward import calculate_reward


class FraudEnvironment:

    def __init__(self, states, labels):
        self.states = np.asarray(states)
        self.labels = np.asarray(labels)

        self.current_index = 0
        self.n_transactions = len(self.states)

    def reset(self):
        """
        Recommence depuis la première transaction.
        """
        self.current_index = 0

        return self.states[self.current_index]

    def step(self, action):
        """
        Exécute une action sur la transaction actuelle.

        action :
            0 = accepter
            1 = bloquer
        """

        real_class = self.labels[self.current_index]

        reward = calculate_reward(
            action,
            real_class
        )

        self.current_index += 1

        # True si toutes les transactions ont été parcourues
        done = self.current_index >= self.n_transactions

        if done:
            next_state = None
        else:
            next_state = self.states[self.current_index]

        return next_state, reward, done