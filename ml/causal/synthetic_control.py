"""Synthetic control method for estimating counterfactual outcomes.

Builds a weighted combination of control units to approximate the
counterfactual trajectory of a treated unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Any, Sequence

try:
    import scipy.optimize as opt

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class SyntheticControlResult:
    unit_id: str
    weights: dict[str, float]
    observed_outcomes: list[float]
    synthetic_outcomes: list[float]
    causal_effect: float
    pre_treatment_rmse: float


class SyntheticControl:
    """Synthetic control estimator.

    Uses constrained optimization (or nearest-neighbor approximation)
    to find weights that minimize pre-treatment RMSE.
    """

    def __init__(self, unit_id: str = "treated"):
        self.unit_id = unit_id
        self._weights: dict[str, float] = {}

    def fit(
        self,
        treated_pre: list[float],
        control_pool: dict[str, list[float]],
        treated_post: list[float] | None = None,
        control_post: dict[str, list[float]] | None = None,
    ) -> SyntheticControlResult:
        """Fit synthetic control weights using pre-treatment data.

        If post-treatment data is provided, estimate causal effect.
        """
        control_ids = list(control_pool.keys())
        n_ctrl = len(control_ids)
        n_pre = len(treated_pre)

        if n_ctrl == 0 or n_pre == 0:
            return SyntheticControlResult(
                unit_id=self.unit_id,
                weights={},
                observed_outcomes=list(treated_pre) + (list(treated_post) if treated_post else []),
                synthetic_outcomes=list(treated_pre) + (list(treated_post) if treated_post else []),
                causal_effect=0.0,
                pre_treatment_rmse=0.0,
            )

        weights_arr = self._optimize_weights(treated_pre, control_pool, control_ids)
        self._weights = {control_ids[i]: weights_arr[i] for i in range(n_ctrl)}

        pre_synthetic = self._weighted_sum(control_pool, weights_arr, n_pre)
        pre_rmse = sqrt(
            mean((treated_pre[i] - pre_synthetic[i]) ** 2 for i in range(n_pre))
        )

        observed: list[float] = list(treated_pre)
        synthetic: list[float] = list(pre_synthetic)
        causal_effect = 0.0

        if treated_post is not None and control_post is not None:
            n_post = len(treated_post)
            observed.extend(treated_post)
            post_synthetic = [
                sum(
                    control_post[cid][t] * weights_arr[i]
                    for i, cid in enumerate(control_ids)
                )
                for t in range(n_post)
            ]
            synthetic.extend(post_synthetic)

            post_diffs = [
                treated_post[t] - post_synthetic[t] for t in range(n_post)
            ]
            causal_effect = mean(post_diffs) if post_diffs else 0.0

        return SyntheticControlResult(
            unit_id=self.unit_id,
            weights=dict(self._weights),
            observed_outcomes=observed,
            synthetic_outcomes=synthetic,
            causal_effect=causal_effect,
            pre_treatment_rmse=pre_rmse,
        )

    def _optimize_weights(
        self,
        treated_pre: list[float],
        control_pool: dict[str, list[float]],
        control_ids: list[str],
    ) -> list[float]:
        n_ctrl = len(control_ids)
        n_pre = len(treated_pre)

        if n_ctrl == 1:
            return [1.0]

        if SCIPY_AVAILABLE:
            return self._optimize_scipy(treated_pre, control_pool, control_ids, n_pre, n_ctrl)

        return self._optimize_grid(treated_pre, control_pool, control_ids, n_pre, n_ctrl)

    def _optimize_scipy(
        self,
        treated_pre: list[float],
        control_pool: dict[str, list[float]],
        control_ids: list[str],
        n_pre: int,
        n_ctrl: int,
    ) -> list[float]:
        def rmse(weights):
            synthetic = self._weighted_sum(control_pool, weights, n_pre)
            return sqrt(
                mean((treated_pre[i] - synthetic[i]) ** 2 for i in range(n_pre))
            )

        x0 = [1.0 / n_ctrl] * n_ctrl
        bounds = [(0.0, 1.0)] * n_ctrl
        constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1.0}]

        result = opt.minimize(
            rmse, x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 1000},
        )

        if result.success:
            return [max(0.0, w) for w in result.x]
        return x0

    def _optimize_grid(
        self,
        treated_pre: list[float],
        control_pool: dict[str, list[float]],
        control_ids: list[str],
        n_pre: int,
        n_ctrl: int,
    ) -> list[float]:
        best_weights = [1.0 / n_ctrl] * n_ctrl
        best_rmse = float("inf")
        step = 0.1

        def _grid_recursive(idx, remaining_weight, current):
            nonlocal best_weights, best_rmse
            if idx == n_ctrl - 1:
                current.append(remaining_weight)
                if remaining_weight < 0:
                    current.pop()
                    return
                synthetic = self._weighted_sum(control_pool, current, n_pre)
                rmse_val = sqrt(
                    mean((treated_pre[i] - synthetic[i]) ** 2 for i in range(n_pre))
                )
                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_weights = list(current)
                current.pop()
            else:
                n_steps = int(round(remaining_weight / step))
                for s in range(n_steps + 1):
                    w = round(s * step, 6)
                    if w > remaining_weight + 1e-9:
                        break
                    current.append(w)
                    _grid_recursive(idx + 1, remaining_weight - w, current)
                    current.pop()

        _grid_recursive(0, 1.0, [])

        total = sum(best_weights)
        if total > 0:
            best_weights = [w / total for w in best_weights]

        return best_weights

    @staticmethod
    def _weighted_sum(
        control_pool: dict[str, list[float]],
        weights: list[float],
        n_time: int,
    ) -> list[float]:
        control_ids = list(control_pool.keys())
        result = [0.0] * n_time
        for i, cid in enumerate(control_ids):
            series = control_pool[cid]
            for t in range(n_time):
                result[t] += weights[i] * series[t] if t < len(series) else 0.0
        return result
