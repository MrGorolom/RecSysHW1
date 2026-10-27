"""Implicit ALS using exact alternating weighted least squares."""

import numpy as np
from scipy.linalg import solve


class IALS:
    def __init__(self, factors=32, reg=1.0, alpha=20.0, iterations=12, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def targets(self, frame):
        return np.ones(len(frame), dtype=np.float64)

    def _pass(self, other, confidence_mask, targets):
        gram = other.T @ other + self.reg * np.eye(self.factors)
        result = np.zeros((confidence_mask.shape[0], self.factors))
        for row in range(confidence_mask.shape[0]):
            lo, hi = confidence_mask.indptr[row:row + 2]
            ids = confidence_mask.indices[lo:hi]
            if not len(ids):
                continue
            factors = other[ids]
            # Targets has identical sparse structure, including explicitly stored zeros.
            q = targets.data[targets.indptr[row]:targets.indptr[row + 1]]
            a = gram + self.alpha * (factors.T @ factors)
            b = factors.T @ ((1 + self.alpha) * q)
            result[row] = solve(a, b, assume_a="pos", check_finite=False)
        return result

    def fit(self, history, frame, metadata=None):
        from data_utils import make_matrix
        target = make_matrix(frame, history.shape, self.targets(frame))
        assert np.array_equal(history.indptr, target.indptr)
        assert np.array_equal(history.indices, target.indices)
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.01, (history.shape[1], self.factors))
        transpose = history.T.tocsr()
        target_t = target.T.tocsr()
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = self._pass(self.item_factors, history, target)
            self.item_factors = self._pass(self.user_factors, transpose, target_t)
            u, v = self.user_factors, self.item_factors
            observed_scores = np.einsum("ij,ij->i", u[frame.user_idx], v[frame.item_idx])
            q = self.targets(frame)
            unknown_base = np.sum((u.T @ u) * (v.T @ v))
            correction = np.sum((1 + self.alpha) * (q - observed_scores) ** 2 - observed_scores ** 2)
            regularizer = self.reg * (np.square(u).sum() + np.square(v).sum())
            self.loss_history.append(float(unknown_base + correction + regularizer))
        return self

    def score(self, user_ids):
        return self.user_factors[user_ids] @ self.item_factors.T
