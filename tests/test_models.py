"""Small analytic checks independent of the full dataset."""

import unittest
import numpy as np
import pandas as pd
import torch

from data_utils import make_matrix
from evaluation import ranking_metrics, topk
from models.item_knn import ItemKNN
from models.ials import IALS
from models.custom_ials import CustomIALS
from models.ease import EASE
from models.fm import FMNetwork, FactorizationMachine


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"user_idx": [0, 0, 1, 1, 2, 2],
                                   "item_idx": [0, 1, 1, 2, 0, 2],
                                   "rating": [5., 1., 4., 3., 2., 5.]})
        self.x = make_matrix(self.frame, (3, 4))
        self.meta = [{"title": "", "brand": "A", "categories": ["games"]}] * 4

    def test_metrics_analytic(self):
        m = ranking_metrics([1, 9, 2], {1, 2}, k=3)
        self.assertAlmostEqual(m["Recall@10"], 1)
        self.assertAlmostEqual(m["MAP@10"], (1 + 2 / 3) / 2)
        self.assertAlmostEqual(m["NDCG@10"], 1.5 / (1 + 1 / np.log2(3)))
        np.testing.assert_array_equal(topk([1., 1., 2., 0.], [2], 2), [0, 1])

    def test_knn_cosine(self):
        model = ItemKNN(neighbors=4).fit(self.x)
        expected = np.full((3, 3), .5)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.similarity.toarray()[:3, :3], expected)
        self.assertTrue(np.isfinite(model.score([0])).all())

    def test_als_exact_update(self):
        model = IALS(factors=2, reg=.3, alpha=4)
        factors = np.array([[.2, .1], [.1, .4], [.5, .2], [.3, .7]])
        actual = model._pass(factors, self.x, self.x)
        for u in range(3):
            p = self.x.toarray()[u]
            c = 1 + 4 * p
            expected = np.linalg.solve(factors.T @ (c[:, None] * factors) + .3 * np.eye(2),
                                       factors.T @ (c * p))
            np.testing.assert_allclose(actual[u], expected)

    def test_custom_target_and_fit(self):
        model = CustomIALS(factors=2, iterations=2)
        q = model.targets(self.frame)
        self.assertGreater(q[0], 0)
        self.assertLess(q[1], 0)
        np.testing.assert_allclose(q[2:4], [.3125, -.0625])
        model.fit(self.x, self.frame)
        self.assertTrue(np.isfinite(model.score([0, 1])).all())

    def test_ease_constrained_solution(self):
        model = EASE(reg=2).fit(self.x)
        gram = self.x.toarray().T @ self.x.toarray()
        for j in range(4):
            ids = np.array([i for i in range(4) if i != j])
            expected = np.linalg.solve(gram[np.ix_(ids, ids)] + 2 * np.eye(3), gram[ids, j])
            np.testing.assert_allclose(model.weights[ids, j], expected, atol=1e-12)
        np.testing.assert_array_equal(np.diag(model.weights), 0)

    def test_fm_formula_and_fast_scoring(self):
        torch.manual_seed(42)
        net = FMNetwork(20, 3)
        ids = torch.tensor([[1, 4, 7, 0], [2, 5, 8, 9]])
        explicit = net.bias + net.linear(ids).sum((1, 2))
        for j in range(ids.shape[1]):
            for k in range(j + 1, ids.shape[1]):
                explicit = explicit + (net.embedding(ids[:, j]) * net.embedding(ids[:, k])).sum(1)
        torch.testing.assert_close(net(ids), explicit)
        fm = FactorizationMachine(factors=3, epochs=1, negatives=1, batch_size=8)
        fm.fit(self.x, self.frame, self.meta)
        all_ids = torch.cat([torch.ones((4, 1), dtype=torch.long), fm.item_features], dim=1)
        with torch.no_grad():
            expected = fm.network(all_ids).numpy()
        np.testing.assert_allclose(fm.score([0])[0], expected, atol=1e-6)

    def test_custom_als_update_including_zero_target(self):
        frame = pd.DataFrame({"user_idx": [0, 0, 0], "item_idx": [0, 1, 2],
                              "rating": [3., 5., 1.]})
        mask = make_matrix(frame, (1, 4))
        model = CustomIALS(factors=2, reg=.3, alpha=4)
        target = make_matrix(frame, mask.shape, model.targets(frame))
        self.assertEqual(target.nnz, mask.nnz)
        self.assertEqual(target.data[0], 0)
        factors = np.array([[.2, .1], [.1, .4], [.5, .2], [.3, .7]])
        confidence = 1 + 4 * mask.toarray()[0]
        expected = np.linalg.solve(
            factors.T @ (confidence[:, None] * factors) + .3 * np.eye(2),
            factors.T @ (confidence * target.toarray()[0]))
        np.testing.assert_allclose(model._pass(factors, mask, target)[0], expected)

    def test_als_objectives_against_dense_formula(self):
        for cls in (IALS, CustomIALS):
            model = cls(factors=2, reg=.3, alpha=4, iterations=3)
            model.fit(self.x, self.frame)
            target = make_matrix(self.frame, self.x.shape, model.targets(self.frame)).toarray()
            confidence = 1 + model.alpha * self.x.toarray()
            scores = model.user_factors @ model.item_factors.T
            expected = np.sum(confidence * (target - scores) ** 2)
            expected += model.reg * (np.square(model.user_factors).sum()
                                     + np.square(model.item_factors).sum())
            self.assertAlmostEqual(model.loss_history[-1], expected, places=9)
            self.assertTrue(np.all(np.diff(model.loss_history) <= 1e-10))

    def test_knn_shrinkage_and_row_neighbors(self):
        model = ItemKNN(neighbors=1, shrinkage=10).fit(self.x)
        expected = np.zeros((4, 4))
        expected[0, 1] = expected[1, 0] = expected[2, 0] = .5 / 11
        np.testing.assert_allclose(model.similarity.toarray(), expected)

    def test_ap_normalization_with_more_than_ten_relevant_items(self):
        metrics = ranking_metrics(list(range(10)), set(range(20)))
        self.assertAlmostEqual(metrics["NDCG@10"], 1)
        self.assertAlmostEqual(metrics["MAP@10"], 1)
        self.assertAlmostEqual(metrics["Recall@10"], .5)
        missed = ranking_metrics(list(range(20, 30)), set(range(20)))
        self.assertEqual(sum(missed.values()), 0)

    def test_bce_formula(self):
        logits = torch.tensor([-1000., -2., 0., 3., 1000.], dtype=torch.float64)
        labels = torch.tensor([0., 1., 0., 1., 1.], dtype=torch.float64)
        expected = np.mean(np.logaddexp(0, logits.numpy()) - labels.numpy() * logits.numpy())
        actual = torch.nn.BCEWithLogitsLoss()(logits, labels).item()
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
