import numpy as np


def build_states(fraud_probabilities, anomaly_scores, amounts):
    states = np.column_stack([
        fraud_probabilities,
        anomaly_scores,
        amounts
    ])

    return states