"""One full-catalog top-k evaluator shared by every model."""

import numpy as np


def ranking_metrics(recommendations, relevant, k=10):
    hits = np.asarray([int(i in relevant) for i in recommendations[:k]], dtype=float)
    n = len(relevant)
    if not n:
        raise ValueError("Metrics require at least one relevant candidate")
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(n, k) + 2))).sum()
    precision_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return {"NDCG@10": float((hits * discounts).sum() / ideal),
            "MAP@10": float((hits * precision_at).sum() / min(n, k)),
            "Recall@10": float(hits.sum() / n)}


def topk(scores, known, k=10):
    """Deterministic tie break by ascending item index."""
    scores = np.asarray(scores, dtype=float).copy()
    scores[known] = -np.inf
    if np.isnan(scores).any():
        raise ValueError("NaN scores")
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) <= k:
        return finite[np.lexsort((finite, -scores[finite]))]
    threshold = np.partition(scores[finite], -k)[-k]
    above = finite[scores[finite] > threshold]
    equal = finite[scores[finite] == threshold][:k - len(above)]
    candidates = np.concatenate([above, equal])
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def evaluate(model, history, holdout, eligible_users, batch_size=128):
    eligible = set(np.asarray(eligible_users).tolist())
    positives = holdout[holdout.rating >= 4].groupby("user_idx").item_idx.apply(list)
    ground_truth = {}
    for u, items in positives.items():
        if u in eligible:
            seen = history.indices[history.indptr[u]:history.indptr[u + 1]]
            relevant = set(items) - set(seen)
            if relevant:
                ground_truth[u] = relevant
    users = np.asarray(sorted(ground_truth), dtype=int)
    if not len(users):
        raise ValueError("No eligible evaluation users")
    per_user = []
    for start in range(0, len(users), batch_size):
        block = users[start:start + batch_size]
        scores = model.score(block)
        for u, row in zip(block, scores):
            known = history.indices[history.indptr[u]:history.indptr[u + 1]]
            recs = topk(row, known)
            assert len(recs) == len(set(recs)) and not set(recs).intersection(known)
            per_user.append({"user_idx": int(u), **ranking_metrics(recs, ground_truth[u])})
    means = {key: float(np.mean([r[key] for r in per_user]))
             for key in ("NDCG@10", "MAP@10", "Recall@10")}
    return {**means, "users": len(users)}, per_user
