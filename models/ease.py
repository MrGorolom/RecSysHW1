"""Closed-form EASE with a zero diagonal constraint."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, history, frame=None, metadata=None):
        self.history = history
        gram = (history.T @ history).toarray().astype(np.float64)
        gram.flat[::gram.shape[0] + 1] += self.reg
        factor = cho_factor(gram, overwrite_a=True, check_finite=False)
        inverse = cho_solve(factor, np.eye(gram.shape[0]), overwrite_b=True, check_finite=False)
        inverse /= -np.diag(inverse).copy()[None, :]
        np.fill_diagonal(inverse, 0)
        self.weights = inverse
        return self

    def score(self, user_ids):
        return self.history[user_ids] @ self.weights
