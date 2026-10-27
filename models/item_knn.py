"""Item-based cosine similarity, implemented with sparse matrix arithmetic."""

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack


class ItemKNN:
    def __init__(self, neighbors=100, shrinkage=0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, history, frame=None, metadata=None):
        self.history = history
        norms = np.sqrt(np.asarray(history.power(2).sum(axis=0)).ravel())
        z = history @ diags(1 / np.maximum(norms, 1e-12))
        blocks = []
        for start in range(0, history.shape[1], 256):
            s = (z[:, start:start + 256].T @ z).tocsr()
            if self.shrinkage:
                co = (history[:, start:start + 256].T @ history).tocsr()
                co.data = co.data / (co.data + self.shrinkage)
                s = s.multiply(co).tocsr()
            rows, cols, values = [], [], []
            for local in range(s.shape[0]):
                left, right = s.indptr[local:local + 2]
                ids, data = s.indices[left:right], s.data[left:right]
                keep = (ids != start + local) & (data > 0)
                ids, data = ids[keep], data[keep]
                order = np.lexsort((ids, -data))[:self.neighbors]
                rows.extend([local] * len(order))
                cols.extend(ids[order])
                values.extend(data[order])
            blocks.append(csr_matrix((values, (rows, cols)), shape=s.shape))
        self.similarity = vstack(blocks, format="csr")
        return self

    def score(self, user_ids):
        return (self.history[user_ids] @ self.similarity).toarray()
