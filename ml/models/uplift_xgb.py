"""Uplift modeling for credit decisions.

Implements a two-model approach to predict the treatment effect of
loan approval on default probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

_HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment,misc]

from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class UpliftResult:
    uplift_scores: list[float]
    treatment_probabilities: list[float]
    control_probabilities: list[float]
    qini_coefficient: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    n_treatment: int = 0
    n_control: int = 0


class UpliftModel:
    """Two-model uplift modeling for credit decisions.

    Predicts the causal effect of approval on default probability.
    Higher (more positive) uplift = worse candidate for approval.
    """

    def __init__(self, treatment_model=None, control_model=None):
        if treatment_model is not None:
            self.treatment_model = treatment_model
        elif _HAS_XGBOOST:
            self.treatment_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
                tree_method="hist",
                verbosity=0,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
            )
        else:
            self.treatment_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

        if control_model is not None:
            self.control_model = control_model
        elif _HAS_XGBOOST:
            self.control_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
                tree_method="hist",
                verbosity=0,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
            )
        else:
            self.control_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

    def fit(
        self,
        treatment_features: list[dict[str, Any]],
        treatment_outcomes: list[int],
        control_features: list[dict[str, Any]],
        control_outcomes: list[int],
    ) -> UpliftResult:
        if not treatment_features or not control_features:
            raise ValueError("treatment_features and control_features must not be empty.")

        treatment_X = _dicts_to_matrix(treatment_features)
        control_X = _dicts_to_matrix(control_features)

        feature_keys = _unify_feature_keys(treatment_features, control_features)
        treatment_X = _align_features(treatment_X, feature_keys)
        control_X = _align_features(control_X, feature_keys)

        self._feature_keys = feature_keys

        self.treatment_model.fit(treatment_X, treatment_outcomes)
        self.control_model.fit(control_X, control_outcomes)

        treatment_probs = [float(p) for p in self.treatment_model.predict_proba(treatment_X)[:, 1]]
        control_probs = [float(p) for p in self.control_model.predict_proba(control_X)[:, 1]]

        uplift_scores = [t - c for t, c in zip(treatment_probs, control_probs)]

        importance = _extract_importance(self.treatment_model, self.control_model, self._feature_keys)

        return UpliftResult(
            uplift_scores=uplift_scores,
            treatment_probabilities=treatment_probs,
            control_probabilities=control_probs,
            qini_coefficient=None,
            feature_importance=importance,
            n_treatment=len(treatment_features),
            n_control=len(control_features),
        )

    def predict_uplift(self, features: list[dict[str, Any]]) -> UpliftResult:
        if not features:
            return UpliftResult(
                uplift_scores=[],
                treatment_probabilities=[],
                control_probabilities=[],
            )

        if not hasattr(self, "_feature_keys") or not self._feature_keys:
            raise RuntimeError("UpliftModel must be fit before predict_uplift.")

        X = _align_features(_dicts_to_matrix(features), self._feature_keys)

        treatment_probs = [float(p) for p in self.treatment_model.predict_proba(X)[:, 1]]
        control_probs = [float(p) for p in self.control_model.predict_proba(X)[:, 1]]

        uplift_scores = [t - c for t, c in zip(treatment_probs, control_probs)]

        return UpliftResult(
            uplift_scores=uplift_scores,
            treatment_probabilities=treatment_probs,
            control_probabilities=control_probs,
            qini_coefficient=None,
            n_treatment=0,
            n_control=0,
        )

    def segment_by_uplift(
        self,
        features: list[dict[str, Any]],
        uplift_scores: list[float],
        n_segments: int = 5,
    ) -> dict[str, Any]:
        if not features or not uplift_scores:
            return {"segments": [], "n_segments": 0}

        sorted_pairs = sorted(zip(uplift_scores, features), key=lambda pair: pair[0])
        segment_size = max(len(sorted_pairs) // n_segments, 1)

        segments = []
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else len(sorted_pairs)
            segment_scores = [s for s, _ in sorted_pairs[start:end]]

            avg_uplift = mean(segment_scores)

            if avg_uplift < -0.05:
                recommendation = "Strongly recommend approval"
            elif avg_uplift < 0.05:
                recommendation = "Use standard risk assessment"
            else:
                recommendation = "Recommend decline"

            segments.append(
                {
                    "segment_index": i,
                    "size": len(segment_scores),
                    "avg_uplift": round(avg_uplift, 6),
                    "min_uplift": round(min(segment_scores), 6),
                    "max_uplift": round(max(segment_scores), 6),
                    "recommendation": recommendation,
                }
            )

        return {"segments": segments, "n_segments": len(segments)}


def _interleaved_perfect_indices(
    outcomes: Sequence[int],
    treatment: Sequence[int],
) -> list[int]:
    """Compute the optimal ordering for maximum cumulative uplift.

    The approach interleaves treatment+event with control+non-event first
    to ensure both groups are present from early steps, then appends
    control+event and treatment+non-event last.
    """
    treat_event: list[int] = []
    treat_non_event: list[int] = []
    control_event: list[int] = []
    control_non_event: list[int] = []

    for i in range(len(outcomes)):
        t = treatment[i]
        o = outcomes[i]
        if t == 1 and o == 1:
            treat_event.append(i)
        elif t == 1 and o == 0:
            treat_non_event.append(i)
        elif t == 0 and o == 1:
            control_event.append(i)
        else:
            control_non_event.append(i)

    result: list[int] = []
    while treat_event or control_non_event:
        if treat_event:
            result.append(treat_event.pop(0))
        if control_non_event:
            result.append(control_non_event.pop(0))
    result.extend(control_event)
    result.extend(treat_non_event)
    return result


def compute_qini_coefficient(
    scores: list[float],
    outcomes: list[int],
    treatment: list[int],
) -> float:
    """Compute Qini coefficient — a Gini-style metric on the uplift curve.

    Qini = (AUUC - AUUR) / (AUUP - AUUR)

    where:
        AUUC = area under the model's uplift curve (sorted by predicted uplift)
        AUUR = area under the random (diagonal) curve
        AUUP = area under the perfect interleaved model's curve

    Returns value in [-1, 1].
    """
    if len(scores) != len(outcomes) or len(scores) != len(treatment):
        raise ValueError("scores, outcomes, and treatment must have the same length.")
    if not scores:
        return 0.0

    n = len(scores)
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

    n_treated = sum(treatment)
    n_control = n - n_treated

    if n_treated == 0 or n_control == 0:
        return 0.0

    total_events_treated = sum(o * t for o, t in zip(outcomes, treatment))
    total_events_control = sum(o * (1 - t) for o, t in zip(outcomes, treatment))

    total_uplift = total_events_treated / n_treated - total_events_control / n_control

    if abs(total_uplift) < 1e-10:
        return 0.0

    def _curve_area(indices: list[int]) -> float:
        cum_t = 0
        cum_c = 0
        t_ev = 0
        c_ev = 0
        area = 0.0
        for idx in indices:
            t = treatment[idx]
            o = outcomes[idx]
            cum_t += t
            cum_c += 1 - t
            t_ev += t * o
            c_ev += (1 - t) * o
            if cum_t > 0 and cum_c > 0:
                area += (t_ev / cum_t) - (c_ev / cum_c)
        return area

    auuc = _curve_area(sorted_indices)

    auur = total_uplift * (n + 1) / 2.0

    perfect_indices = _interleaved_perfect_indices(outcomes, treatment)
    auup = _curve_area(perfect_indices)

    denom = auup - auur
    if abs(denom) < 1e-10:
        return 0.0

    qini = (auuc - auur) / denom
    return max(-1.0, min(1.0, qini))


def uplift_decision_rule(uplift_score: float, threshold: float = 0.0) -> str:
    if uplift_score < -0.05:
        return "Strongly recommend approval"
    if uplift_score > 0.05:
        return "Recommend decline"
    return "Use standard risk assessment"


def _dicts_to_matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    if not rows:
        return []
    feature_keys = sorted(rows[0].keys())
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([float(row.get(key, 0.0)) for key in feature_keys])
    return matrix


def _unify_feature_keys(
    treatment_features: list[dict[str, Any]],
    control_features: list[dict[str, Any]],
) -> list[str]:
    keys: set[str] = set()
    for row in treatment_features:
        keys.update(row.keys())
    for row in control_features:
        keys.update(row.keys())
    return sorted(keys)


def _align_features(
    matrix: list[list[float]],
    feature_keys: list[str],
) -> list[list[float]]:
    return matrix


def _extract_importance(
    treatment_model: Any,
    control_model: Any,
    feature_keys: list[str],
) -> dict[str, float]:
    importances: dict[str, float] = {}

    for label, model in (("treatment", treatment_model), ("control", control_model)):
        if hasattr(model, "feature_importances_"):
            raw = model.feature_importances_
            for idx, key in enumerate(feature_keys):
                if idx < len(raw):
                    val = float(raw[idx])
                    existing = importances.get(key, 0.0)
                    importances[key] = max(existing, val)

    return dict(sorted(importances.items(), key=lambda item: -item[1]))
