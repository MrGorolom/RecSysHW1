"""Second-order FM, trained on observed and sampled unobserved pairs."""

import numpy as np
import torch
from torch import nn


class FMNetwork(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        self.embedding = nn.Embedding(n_features, factors, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, ids):
        v = self.embedding(ids)
        pairwise = 0.5 * (v.sum(1).square() - v.square().sum(1)).sum(1)
        return self.bias + self.linear(ids).sum((1, 2)) + pairwise


class FactorizationMachine:
    def __init__(self, factors=32, epochs=8, lr=0.003, weight_decay=1e-5,
                 negatives=4, batch_size=4096, metadata=True, seed=42):
        self.factors, self.epochs, self.lr = factors, epochs, lr
        self.weight_decay, self.negatives = weight_decay, negatives
        self.batch_size, self.use_metadata, self.seed = batch_size, metadata, seed

    def _features(self, metadata, n_users, n_items):
        # 0 is padding; each nonzero id belongs to one distinct feature.
        vocabulary = {}
        offset = 1 + n_users + n_items
        rows = []
        for i, meta in enumerate(metadata):
            tokens = []
            if self.use_metadata:
                tokens = ["brand:" + (meta["brand"] or "UNKNOWN")]
                tokens += ["category:" + c for c in meta["categories"]]
                if not meta["categories"]:
                    tokens.append("category:UNKNOWN")
            row = [1 + n_users + i]
            for token in sorted(set(tokens)):
                if token not in vocabulary:
                    vocabulary[token] = offset + len(vocabulary)
                row.append(vocabulary[token])
            rows.append(row)
        features = np.zeros((n_items, max(map(len, rows))), dtype=np.int64)
        for i, row in enumerate(rows):
            features[i, :len(row)] = row
        return torch.from_numpy(features), offset + len(vocabulary)

    def fit(self, history, frame, metadata):
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(self.seed)
        n_users, n_items = history.shape
        self.n_users = n_users
        self.item_features, n_features = self._features(metadata, n_users, n_items)
        self.network = FMNetwork(n_features, self.factors)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        pos_u, pos_i = frame.user_idx.to_numpy(), frame.item_idx.to_numpy()
        known = np.sort(pos_u.astype(np.int64) * n_items + pos_i)
        self.loss_history = []
        for epoch in range(self.epochs):
            neg_u = np.repeat(pos_u, self.negatives)
            neg_i = rng.integers(0, n_items, len(neg_u))
            # Reject every known training pair, without inspecting future data.
            bad = np.isin(neg_u * n_items + neg_i, known, assume_unique=False)
            while bad.any():
                neg_i[bad] = rng.integers(0, n_items, int(bad.sum()))
                bad = np.isin(neg_u * n_items + neg_i, known, assume_unique=False)
            users = np.concatenate([pos_u, neg_u])
            items = np.concatenate([pos_i, neg_i])
            labels = np.concatenate([np.ones(len(pos_u)), np.zeros(len(neg_u))]).astype(np.float32)
            order = rng.permutation(len(users))
            total = 0.0
            self.network.train()
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start + self.batch_size]
                u = torch.from_numpy(users[ix] + 1)
                i = torch.from_numpy(items[ix])
                ids = torch.cat([u[:, None], self.item_features[i]], dim=1)
                y = torch.from_numpy(labels[ix])
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.network(ids), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(ix)
            self.loss_history.append(total / len(users))
        # Exact decomposition of FM into user/item terms, avoiding all pair materialization.
        self.network.eval()
        with torch.no_grad():
            v = self.network.embedding(self.item_features)
            self.item_vectors = v.sum(1)
            item_interactions = 0.5 * (self.item_vectors.square() - v.square().sum(1)).sum(1)
            self.item_bias = self.network.linear(self.item_features).sum((1, 2)) + item_interactions
        return self

    def score(self, user_ids):
        with torch.no_grad():
            u = torch.as_tensor(np.asarray(user_ids) + 1, dtype=torch.long)
            scores = self.network.embedding(u) @ self.item_vectors.T
            scores += self.item_bias[None, :] + self.network.linear(u) + self.network.bias
            return scores.numpy()
