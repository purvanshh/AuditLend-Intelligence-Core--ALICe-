"""Propensity Score Matching for causal inference in credit decisions.

Estimates the Average Treatment Effect on the Treated (ATT) for
credit-limit increases and other lending policy changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from statistics import mean, stdev
from typing import Any, Callable, Sequence

try:
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class MatchedPair:
    treatment_id: str
    control_id: str
    propensity_treatment: float
    propensity_control: float
    weight: float = 1.0


@dataclass
class PSMResult:
    matched_pairs: list[MatchedPair]
    treatment_outcomes: list[float]
    control_outcomes: list[float]
    att: float
    att_ci: tuple[float, float]
    balance_statistics: dict[str, dict[str, float]]
    n_treatment: int
    n_control: int
    n_matched: int


class PropensityScoreMatcher:
    """Propensity Score Matching for causal inference.

    Uses logistic regression (via sklearn, optional) or a simple
    dot-product scoring function to estimate propensity scores.
    Supports nearest-neighbor matching with caliper.
    """

    def __init__(self, caliper: float = 0.05, n_neighbors: int = 1):
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self._propensity_scores: list[float] = []

    def fit_propensity(
        self,
        features: list[dict[str, float]],
        treatment: list[int],
        scoring_fn: Callable | None = None,
    ) -> list[float]:
        """Estimate propensity scores P(treatment=1 | features).

        If scoring_fn is provided, use it directly.
        If sklearn is available, fit LogisticRegression.
        Otherwise, use a simple distance-based heuristic.
        """
        if not features or not treatment:
            self._propensity_scores = [0.5] * len(features) if features else []
            return self._propensity_scores

        if scoring_fn is not None:
            scores = [scoring_fn(f) for f in features]
            self._propensity_scores = scores
            return self._propensity_scores

        feature_names = sorted(features[0].keys())
        X = [[f.get(k, 0.0) for k in feature_names] for f in features]
        y = list(treatment)

        if SKLEARN_AVAILABLE:
            model = LogisticRegression(
                penalty=None, solver="lbfgs", max_iter=1000, random_state=42
            )
            model.fit(X, y)
            scores = [float(p[1]) for p in model.predict_proba(X)]
            self._propensity_scores = scores
            return self._propensity_scores

        treated = [X[i] for i in range(len(X)) if y[i] == 1]
        control = [X[i] for i in range(len(X)) if y[i] == 0]

        if not treated or not control:
            self._propensity_scores = [0.5] * len(X)
            return self._propensity_scores

        treated_mean = [mean(col) for col in zip(*treated)]
        n_feats = len(feature_names)

        scores = []
        for x in X:
            dist = sqrt(sum((x[j] - treated_mean[j]) ** 2 for j in range(n_feats)))
            scores.append(1.0 / (1.0 + dist))

        self._propensity_scores = scores
        return self._propensity_scores

    def match(
        self,
        treatment_features: list[dict],
        control_features: list[dict],
        treatment_outcomes: list[float],
        control_outcomes: list[float],
        treatment_ids: list[str] | None = None,
        control_ids: list[str] | None = None,
    ) -> PSMResult:
        """Match treated units to control units by propensity score."""
        t_features = list(treatment_features)
        c_features = list(control_features)
        t_outcomes = list(treatment_outcomes)
        c_outcomes = list(control_outcomes)

        n_treat = len(t_features)
        n_ctrl = len(c_features)

        if treatment_ids is None:
            treatment_ids = [f"treatment_{i}" for i in range(n_treat)]
        if control_ids is None:
            control_ids = [f"control_{i}" for i in range(n_ctrl)]

        all_features = t_features + c_features
        all_treatment = [1] * n_treat + [0] * n_ctrl
        self.fit_propensity(all_features, all_treatment)

        t_scores = self._propensity_scores[:n_treat]
        c_scores = self._propensity_scores[n_treat:]

        matches = self._nearest_neighbor_match(t_scores, c_scores)

        matched_pairs: list[MatchedPair] = []
        t_matched_outcomes: list[float] = []
        c_matched_outcomes: list[float] = []

        for t_idx, c_idx, dist in matches:
            matched_pairs.append(
                MatchedPair(
                    treatment_id=treatment_ids[t_idx],
                    control_id=control_ids[c_idx],
                    propensity_treatment=t_scores[t_idx],
                    propensity_control=c_scores[c_idx],
                    weight=1.0,
                )
            )
            t_matched_outcomes.append(t_outcomes[t_idx])
            c_matched_outcomes.append(c_outcomes[c_idx])

        att_val = 0.0
        att_ci = (0.0, 0.0)
        if t_matched_outcomes and c_matched_outcomes:
            diffs = [t - c for t, c in zip(t_matched_outcomes, c_matched_outcomes)]
            att_val = mean(diffs)
            if len(diffs) > 1:
                se = stdev(diffs) / sqrt(len(diffs))
                att_ci = (att_val - 1.96 * se, att_val + 1.96 * se)

        t_matched_features = [t_features[i] for i, _, _ in matches]
        c_matched_features = [c_features[j] for _, j, _ in matches]
        balance = compute_balance(t_matched_features, c_matched_features)

        return PSMResult(
            matched_pairs=matched_pairs,
            treatment_outcomes=t_matched_outcomes,
            control_outcomes=c_matched_outcomes,
            att=att_val,
            att_ci=att_ci,
            balance_statistics=balance,
            n_treatment=n_treat,
            n_control=n_ctrl,
            n_matched=len(matched_pairs),
        )

    def _nearest_neighbor_match(
        self,
        treatment_scores: list[float],
        control_scores: list[float],
    ) -> list[tuple[int, int, float]]:
        """One-to-one nearest-neighbor matching within caliper."""
        available = set(range(len(control_scores)))
        matches: list[tuple[int, int, float]] = []

        treated_indices = sorted(
            range(len(treatment_scores)), key=lambda i: treatment_scores[i], reverse=True
        )

        for t_idx in treated_indices:
            t_score = treatment_scores[t_idx]
            best_c_idx = -1
            best_dist = float("inf")

            for c_idx in sorted(available):
                dist = abs(t_score - control_scores[c_idx])
                if dist < best_dist and dist <= self.caliper:
                    best_dist = dist
                    best_c_idx = c_idx

            if best_c_idx != -1:
                matches.append((t_idx, best_c_idx, best_dist))
                if self.n_neighbors <= 1:
                    available.remove(best_c_idx)

        return matches


def compute_balance(
    treatment_features: list[dict], control_features: list[dict]
) -> dict[str, dict[str, float]]:
    """Compute standardized mean differences for covariate balance check.

    Returns per-feature: std_diff, var_ratio, before/after comparison.
    """
    if not treatment_features or not control_features:
        return {}

    feature_names = sorted(treatment_features[0].keys())
    result: dict[str, dict[str, float]] = {}

    for feat in feature_names:
        t_vals = [f.get(feat, 0.0) for f in treatment_features]
        c_vals = [f.get(feat, 0.0) for f in control_features]

        t_mean = mean(t_vals)
        c_mean = mean(c_vals)
        t_var = stdev(t_vals) ** 2 if len(t_vals) > 1 else 0.0
        c_var = stdev(c_vals) ** 2 if len(c_vals) > 1 else 0.0

        pooled_sd = sqrt((t_var + c_var) / 2.0) if (t_var + c_var) > 0 else 1.0
        std_diff = (t_mean - c_mean) / pooled_sd
        var_ratio = t_var / c_var if c_var > 0 else float("inf")

        result[feat] = {
            "std_diff": round(std_diff, 6),
            "var_ratio": round(var_ratio, 6),
            "mean_treatment": round(t_mean, 6),
            "mean_control": round(c_mean, 6),
        }

    return result
