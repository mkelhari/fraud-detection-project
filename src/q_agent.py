import numpy as np
import random


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.0,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Probabilité d'explorer
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Dictionnaire contenant les valeurs Q
        self.q_table = {}

    def discretize_state(self, state):
        """
        Transforme l'état continu en état discret.
        """

        fraud_probability = state[0]
        anomaly_score = state[1]
        amount = state[2]

        # Discrétisation
        fraud_bin = int(fraud_probability * 10)

        anomaly_bin = int(
            np.clip((anomaly_score + 0.5) * 10, 0, 10)
        )

        amount_bin = int(
            np.clip((amount + 5) * 2, 0, 20)
        )

        return (fraud_bin, anomaly_bin, amount_bin)

    def get_q_values(self, state):

        discrete_state = self.discretize_state(state)

        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(2)

        return self.q_table[discrete_state]

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.choice([0, 1])

        # Exploitation
        q_values = self.get_q_values(state)

        return np.argmax(q_values)

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        current_q_values = self.get_q_values(state)

        current_q = current_q_values[action]

        if done:
            target = reward

        else:
            next_q_values = self.get_q_values(next_state)

            target = reward + (
                self.discount_factor
                * np.max(next_q_values)
            )

        # Formule Q-learning
        new_q = current_q + self.learning_rate * (
            target - current_q
        )

        current_q_values[action] = new_q

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def predict_action(self, state):
        """
        Choisit la meilleure action sans modifier la Q-table.

        Si l'état n'a jamais été rencontré pendant
        l'apprentissage, on utilise la probabilité
        du Random Forest comme solution de secours.
        """

        discrete_state = self.discretize_state(state)

        if discrete_state not in self.q_table:

            fraud_probability = state[0]

            if fraud_probability >= 0.21:
                return 1  # bloquer
            else:
                return 0  # accepter

        q_values = self.q_table[discrete_state]

        return int(np.argmax(q_values))
