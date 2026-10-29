"""CPU numerical contracts for the native gradient optimizer (no model downloads)."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_optimize import optimize_style


class NativeOptimizerTests(unittest.TestCase):
    def test_true_gradients_reduce_full_objective_and_keep_unit_rows(self):
        initial = torch.tensor([[[0., 1.], [0., 1.]]])
        target = torch.tensor([[[1., 0.], [1., 0.]]])
        bests = []
        def objective(x, prompt):
            return ((x - target) ** 2).mean() * (prompt + 1)
        best, report = optimize_style(initial, objective, 3, iterations=80, threshold=0.001,
                                     rate=0.05, on_best=lambda x, e: bests.append((x.clone(), e)))
        self.assertLess(report["best_loss"], 0.01)
        self.assertGreater(report["backward_calls"], 0)
        self.assertTrue(torch.allclose(best.norm(dim=-1), torch.ones(1, 2), atol=1e-6))
        self.assertAlmostEqual(sum(objective(best, i).item() for i in range(3)) / 3, report["best_loss"])
        self.assertTrue(all(a[1]["best_loss"] > b[1]["best_loss"] for a, b in zip(bests, bests[1:])))
        self.assertTrue(torch.equal(initial, torch.tensor([[[0., 1.], [0., 1.]]])))

    def test_misleading_training_prompt_cannot_accept_or_stop(self):
        initial = torch.tensor([[[0., 1.]]])
        def objective(x, prompt):
            return 1 - x[..., 0].mean() if prompt == 0 else 1 + 4 * x[..., 0].mean()
        best, report = optimize_style(initial, objective, 2, iterations=1, threshold=0.24,
                                     validation_interval=5, rate=0.1, recovery_patience=1)
        self.assertTrue(torch.equal(best, initial))
        self.assertEqual(report["best_loss"], 1)
        self.assertEqual(report["recoveries"], 1)
        self.assertEqual(report["status"], "iteration limit")
        self.assertEqual(report["evaluations"], 5)

    def test_threshold_is_full_mean_and_initial_target_needs_no_backward(self):
        best, report = optimize_style(torch.tensor([[[1., 0.]]]), lambda x, i: x.sum() * 0 + 0.2, 2)
        self.assertEqual(report["status"], "threshold reached")
        self.assertEqual(report["backward_calls"], 0)

    def test_lower_custom_target_does_not_stop_at_the_old_default(self):
        _, report = optimize_style(torch.tensor([[[1., 0.]]]),
            lambda x, i: x.sum() * 0 + 0.2, 2, threshold=0.05, iterations=1)
        self.assertEqual(report["target_loss"], 0.05)
        self.assertEqual(report["backward_calls"], 1)
        self.assertEqual(report["status"], "iteration limit")
        self.assertGreater(report["best_loss"], report["target_loss"])
        for target in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                optimize_style(torch.tensor([[[1., 0.]]]), lambda x, i: x.sum(), 1, threshold=target)

    def test_cancel_retains_published_best_and_never_returns_unvalidated_candidate(self):
        saved = []
        def on_step(e):
            if e["step"] == 2:
                raise KeyboardInterrupt()
        initial = torch.tensor([[[0., 1.]]])
        best, report = optimize_style(initial, lambda x, i: 1 - x[..., 0].mean(), 2,
            iterations=0, validation_interval=5, on_step=on_step, on_best=lambda v, e: saved.append(v.clone()))
        self.assertEqual(report["status"], "cancelled")
        self.assertTrue(torch.equal(best, initial))
        self.assertTrue(torch.equal(best, saved[-1]))

    def test_invalid_gradient_and_nonfinite_loss_fail(self):
        initial = torch.tensor([[[0., 1.]]])
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            optimize_style(initial, lambda x, i: x.sum() * float('nan'), 1, iterations=1)
        with self.assertRaises(ValueError):
            optimize_style(initial, lambda x, i: x.sum(), 1, validation_interval=0)


if __name__ == "__main__":
    unittest.main()
