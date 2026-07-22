"""Cox Proportional Hazards model for credit risk survival analysis.

Models the hazard rate as a function of borrower covariates:
    h(t|X) = h_0(t) * exp(beta_1*X_1 + ... + beta_p*X_p)

Hazard ratios (exp(beta)) quantify the multiplicative effect of each
covariate on the instantaneous default risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class CoxPHResult:
    """Output of a Cox Proportional Hazards fit."""

    coefficients: dict[str, float]
    hazard_ratios: dict[str, float]
    standard_errors: dict[str, float]
    z_scores: dict[str, float]
    p_values: dict[str, float]
    concordance_index: float | None = None
    baseline_hazard: dict[float, float] = field(default_factory=dict)
    n_observations: int = 0
    n_events: int = 0
    n_features: int = 0

    def summary(self) -> str:
        """Return a formatted regression summary."""
        lines = [
            "Cox Proportional Hazards Regression",
            "=" * 60,
            f"  Observations: {self.n_observations}",
            f"  Events: {self.n_events}",
            f"  Features: {self.n_features}",
            f"  Concordance Index: {self.concordance_index:.4f}" if self.concordance_index else "",
            "",
            f"{'Feature':30s} {'coef':>8s} {'exp(coef)':>10s} {'se(coef)':>9s} {'z':>8s} {'p':>8s}",
            "-" * 75,
        ]
        for feature in sorted(self.coefficients.keys()):
            coef = self.coefficients.get(feature, 0.0)
            hr = self.hazard_ratios.get(feature, 1.0)
            se = self.standard_errors.get(feature, 0.0)
            z = self.z_scores.get(feature, 0.0)
            p = self.p_values.get(feature, 1.0)
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            lines.append(
                f"{feature:30s} {coef:>8.4f} {hr:>10.4f} {se:>9.4f} {z:>8.2f} {p_str:>8s}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class CoxPHFitter:
    """Cox Proportional Hazards model for time-to-default data.

    Uses Breslow approximation for tied event times. Fits via Newton-Raphson
    optimization of the partial likelihood.

    Usage:
        fitter = CoxPHFitter()
        result = fitter.fit(df, duration_col='duration_months',
                            event_col='defaulted', feature_cols=['dti_ratio', 'interest_rate_pct'])
        print(result.summary())
        print(f"Hazard ratio for DTI: {result.hazard_ratios['dti_ratio']:.2f}")
    """

    def __init__(self, alpha: float = 0.05, max_iter: int = 100, tol: float = 1e-7):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        rows: list[dict[str, Any]],
        duration_col: str = "duration_months",
        event_col: str = "target_defaulted",
        feature_cols: list[str] | None = None,
    ) -> CoxPHResult:
        """Fit the Cox model.

        Args:
            rows: List of observation rows containing duration, event, and feature values.
            duration_col: Column name for observed time.
            event_col: Column name for event indicator (1=default, 0=censored).
            feature_cols: Subset of features to use. If None, uses all numeric
                         columns except duration and event.

        Returns:
            CoxPHResult with coefficients, hazard ratios, and statistics.
        """
        if not rows:
            return CoxPHResult(
                coefficients={}, hazard_ratios={}, standard_errors={},
                z_scores={}, p_values={},
            )

        available_features = [k for k in (rows[0] or {}).keys()
                              if k not in (duration_col, event_col, "loan_id", "issue_date",
                                           "split", "grade", "sub_grade", "purpose",
                                           "home_ownership", "verification_status", "loan_status")
                              and isinstance(rows[0].get(k), (int, float))]

        features = feature_cols or available_features

        X = []
        durations = []
        events = []
        for row in rows:
            X.append([float(row.get(f, 0.0)) for f in features])
            durations.append(float(row.get(duration_col, 0)))
            events.append(int(bool(row.get(event_col, 0))))

        n = len(rows)
        p = len(features)
        if n == 0 or p == 0:
            return CoxPHResult(
                coefficients={f: 0.0 for f in features},
                hazard_ratios={f: 1.0 for f in features},
                standard_errors={f: 0.0 for f in features},
                z_scores={f: 0.0 for f in features},
                p_values={f: 1.0 for f in features},
            )

        beta = [0.0] * p
        for iteration in range(self.max_iter):
            grad = self._gradient(X, durations, events, beta)
            hessian = self._hessian(X, durations, events, beta)

            if all(abs(g) < self.tol for g in grad):
                break

            hessian_inv = self._invert_matrix(hessian, p)
            step = [sum(hessian_inv[i][j] * grad[j] for j in range(p)) for i in range(p)]

            beta = [beta[i] + step[i] for i in range(p)]

        var_cov = self._invert_matrix(
            self._hessian(X, durations, events, beta), p
        )
        se = [sqrt(abs(var_cov[i][i])) if len(var_cov) > i and len(var_cov[i]) > i else 0.0
              for i in range(p)]
        z_scores = [beta[i] / se[i] if se[i] > 0 else 0.0 for i in range(p)]
        p_values = [self._p_from_z(z) for z in z_scores]

        coefficients = {features[i]: beta[i] for i in range(p)}
        hazard_ratios = {features[i]: exp(beta[i]) for i in range(p)}
        standard_errors = {features[i]: se[i] for i in range(p)}
        z_scores_dict = {features[i]: z_scores[i] for i in range(p)}
        p_values_dict = {features[i]: p_values[i] for i in range(p)}

        concordance = self._concordance_index(X, durations, events, beta)
        baseline_hazard = self._compute_baseline_hazard(X, durations, events, beta)
        n_events = sum(events)

        return CoxPHResult(
            coefficients=coefficients,
            hazard_ratios=hazard_ratios,
            standard_errors=standard_errors,
            z_scores=z_scores_dict,
            p_values=p_values_dict,
            concordance_index=concordance,
            baseline_hazard=baseline_hazard,
            n_observations=n,
            n_events=n_events,
            n_features=p,
        )

    def predict_risk_score(self, features: dict[str, float],
                           result: CoxPHResult) -> float:
        """Compute the linear predictor (risk score) for a single observation.

        risk_score = sum(beta_i * X_i)

        Higher scores indicate higher hazard (default risk).
        """
        score = 0.0
        for feature, coef in result.coefficients.items():
            score += coef * features.get(feature, 0.0)
        return score

    def predict_hazard_ratio(self, features: dict[str, float],
                             result: CoxPHResult) -> float:
        """Compute the hazard ratio relative to the baseline.

        HR = exp(sum(beta_i * X_i))

        A hazard ratio of 2.0 means the subject's default risk is 2x the
        baseline at any given time.
        """
        return exp(self.predict_risk_score(features, result))

    def _gradient(self, X, durations, events, beta):
        p = len(beta)
        grad = [0.0] * p
        order = sorted(range(len(durations)), key=lambda i: durations[i])
        risk_set_sum_exp = [0.0] * p
        n_in_risk_set = 0

        for idx in reversed(order):
            exp_xb = exp(sum(beta[j] * X[idx][j] for j in range(p)))
            for j in range(p):
                risk_set_sum_exp[j] += X[idx][j] * exp_xb
            n_in_risk_set += 1

            if events[idx]:
                for j in range(p):
                    grad[j] += X[idx][j] - risk_set_sum_exp[j] / (n_in_risk_set + 1e-10)

        return grad

    def _hessian(self, X, durations, events, beta):
        p = len(beta)
        hessian = [[0.0] * p for _ in range(p)]
        order = sorted(range(len(durations)), key=lambda i: durations[i])
        risk_set_sum_exp = [0.0] * p
        risk_set_sum_exp2 = [[0.0] * p for _ in range(p)]
        n_in_risk_set = 0

        for idx in reversed(order):
            exp_xb = exp(sum(beta[j] * X[idx][j] for j in range(p)))
            for j in range(p):
                risk_set_sum_exp[j] += X[idx][j] * exp_xb
                for k in range(p):
                    risk_set_sum_exp2[j][k] += X[idx][j] * X[idx][k] * exp_xb
            n_in_risk_set += 1

            if events[idx]:
                denom = n_in_risk_set + 1e-10
                for j in range(p):
                    for k in range(p):
                        hessian[j][k] -= (
                            risk_set_sum_exp2[j][k] / denom
                            - (risk_set_sum_exp[j] / denom) * (risk_set_sum_exp[k] / denom)
                        )
        return hessian

    def _invert_matrix(self, matrix, n):
        if n == 0:
            return []
        augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
                     for i, row in enumerate(matrix)]
        for i in range(n):
            pivot = augmented[i][i]
            if abs(pivot) < 1e-12:
                continue
            for j in range(2 * n):
                augmented[i][j] /= pivot
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]
        return [[row[n + j] for j in range(n)] for row in augmented]

    def _concordance_index(self, X, durations, events, beta):
        concordant = 0
        discordant = 0
        comparable = 0
        n = len(durations)

        for i in range(n):
            if not events[i]:
                continue
            for j in range(n):
                if i == j or not events[j]:
                    continue
                if abs(durations[i] - durations[j]) < 1e-10:
                    continue
                comparable += 1
                risk_i = sum(beta[k] * X[i][k] for k in range(len(beta)))
                risk_j = sum(beta[k] * X[j][k] for k in range(len(beta)))
                if durations[i] < durations[j] and risk_i > risk_j:
                    concordant += 1
                elif durations[i] < durations[j] and risk_i < risk_j:
                    discordant += 1
                elif durations[j] < durations[i] and risk_j > risk_i:
                    concordant += 1
                elif durations[j] < durations[i] and risk_j < risk_i:
                    discordant += 1

        if comparable == 0:
            return 0.5
        return concordant / comparable

    def _compute_baseline_hazard(self, X, durations, events, beta):
        baseline = {}
        order = sorted(range(len(durations)), key=lambda i: durations[i])
        at_risk = len(durations)
        cumulative_hazard = 0.0

        for idx in order:
            risk_score = sum(beta[j] * X[idx][j] for j in range(len(beta)))
            if events[idx] and at_risk > 0:
                hazard_increment = 1.0 / (at_risk * exp(risk_score) + 1e-10)
                cumulative_hazard += hazard_increment
                baseline[durations[idx]] = cumulative_hazard
            at_risk -= 1

        return baseline

    def _p_from_z(self, z):
        abs_z = abs(z)
        if abs_z < 0.5:
            return 0.5
        if abs_z < 1.0:
            return 0.3
        if abs_z < 1.5:
            return 0.13
        if abs_z < 1.96:
            return 0.05
        if abs_z < 2.5:
            return 0.012
        if abs_z < 3.0:
            return 0.003
        if abs_z < 3.5:
            return 0.0005
        return 0.0001
