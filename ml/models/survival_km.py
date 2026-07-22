"""Kaplan-Meier survival analysis for credit risk.

Estimates time-to-default survival functions stratified by borrower
characteristics (grade, purpose, verification status). Produces survival
curves with confidence intervals for portfolio-level hazard assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SurvivalCurve:
    """A survival curve at discrete time points."""

    times: list[float]
    survival_probabilities: list[float]
    confidence_intervals_lower: list[float] | None = None
    confidence_intervals_upper: list[float] | None = None

    def hazard_at_time(self, t: float) -> float:
        """Compute instantaneous hazard rate at time t."""
        for i, time_point in enumerate(self.times):
            if time_point >= t:
                if i == 0:
                    return 1.0 - self.survival_probabilities[0]
                prev_survival = self.survival_probabilities[i - 1]
                if prev_survival == 0:
                    return 0.0
                return 1.0 - self.survival_probabilities[i] / prev_survival
        return 0.0

    def median_survival_time(self) -> float | None:
        """Return the first time at which survival drops below 0.5."""
        for t, s in zip(self.times, self.survival_probabilities):
            if s <= 0.5:
                return t
        return None

    def survival_at_time(self, t: float) -> float:
        """Return the survival probability at or just after time t."""
        for time_point, survival in zip(self.times, self.survival_probabilities):
            if time_point >= t:
                return survival
        return self.survival_probabilities[-1] if self.survival_probabilities else 1.0


@dataclass(frozen=True)
class KaplanMeierResult:
    """Result of a Kaplan-Meier estimation, potentially with multiple strata."""

    strata: dict[str, SurvivalCurve]
    overall_curve: SurvivalCurve | None = None
    strata_counts: dict[str, int] = field(default_factory=dict)
    strata_event_counts: dict[str, int] = field(default_factory=dict)


class KaplanMeierFitter:
    """Kaplan-Meier estimator for time-to-default data.

    Usage:
        fitter = KaplanMeierFitter()
        result = fitter.fit(times, events)
        curve = result.overall_curve

    For stratified analysis:
        strata_result = fitter.fit_stratified(df, duration_col='duration_months',
                                               event_col='defaulted', strata_col='grade')
    """

    def fit(
        self,
        durations: list[float],
        events: list[bool | int],
        confidence_level: float = 0.95,
    ) -> KaplanMeierResult:
        """Fit the Kaplan-Meier estimator on a single population.

        Args:
            durations: Observed times (months to event or censoring).
            events: True/1 if event (default) occurred, False/0 if censored.
            confidence_level: Confidence level for Greenwood confidence intervals.

        Returns:
            KaplanMeierResult with overall survival curve.
        """
        paired = sorted(
            [(d, int(bool(e))) for d, e in zip(durations, events)],
            key=lambda x: x[0],
        )
        n = len(paired)
        if n == 0:
            return KaplanMeierResult(strata={})

        at_risk = n
        survival = 1.0
        times: list[float] = []
        survival_probs: list[float] = []
        var_list: list[float] = []

        z = _z_score(confidence_level)

        i = 0
        while i < n:
            current_time = paired[i][0]
            events_at_t = 0
            censored_at_t = 0
            j = i
            while j < n and paired[j][0] == current_time:
                if paired[j][1] == 1:
                    events_at_t += 1
                else:
                    censored_at_t += 1
                j += 1

            if events_at_t > 0:
                hazard = events_at_t / at_risk
                survival *= 1.0 - hazard
                var_term = events_at_t / (at_risk * (at_risk - events_at_t)) if at_risk > events_at_t else 0.0
                var_list.append(var_term)

            times.append(current_time)
            survival_probs.append(survival)

            at_risk -= (events_at_t + censored_at_t)
            i = j

        cumulative_var = [sum(var_list[: k + 1]) for k in range(len(var_list))]
        std_errors = [v ** 0.5 * s for v, s in zip(cumulative_var, survival_probs)]
        lower = [max(0.0, s - z * se) for s, se in zip(survival_probs, std_errors)]
        upper = [min(1.0, s + z * se) for s, se in zip(survival_probs, std_errors)]

        curve = SurvivalCurve(
            times=times,
            survival_probabilities=survival_probs,
            confidence_intervals_lower=lower,
            confidence_intervals_upper=upper,
        )
        total_events = sum(events)
        return KaplanMeierResult(
            strata={"all": curve},
            overall_curve=curve,
            strata_counts={"all": n},
            strata_event_counts={"all": total_events},
        )

    def fit_stratified(
        self,
        rows: list[dict[str, Any]],
        duration_col: str = "duration_months",
        event_col: str = "target_defaulted",
        strata_col: str = "grade",
    ) -> KaplanMeierResult:
        """Fit Kaplan-Meier separately for each stratum.

        Args:
            rows: List of feature/observation rows.
            duration_col: Column name for observed duration.
            event_col: Column name for event indicator.
            strata_col: Column name for strata assignment.

        Returns:
            KaplanMeierResult with per-stratum survival curves.
        """
        strata_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(strata_col, "UNKNOWN"))
            if key not in strata_groups:
                strata_groups[key] = []
            strata_groups[key].append(row)

        curves: dict[str, SurvivalCurve] = {}
        counts: dict[str, int] = {}
        event_counts: dict[str, int] = {}

        for stratum, group_rows in sorted(strata_groups.items()):
            durations = [float(r.get(duration_col, 0)) for r in group_rows]
            events = [bool(r.get(event_col, 0)) for r in group_rows]
            result = self.fit(durations, events)
            if result.overall_curve is not None:
                curves[stratum] = result.overall_curve
            counts[stratum] = len(group_rows)
            event_counts[stratum] = sum(events)

        return KaplanMeierResult(
            strata=curves,
            overall_curve=None,
            strata_counts=counts,
            strata_event_counts=event_counts,
        )


def _z_score(confidence_level: float) -> float:
    """Approximate z-score for a given confidence level."""
    if confidence_level >= 0.99:
        return 2.576
    if confidence_level >= 0.95:
        return 1.96
    if confidence_level >= 0.90:
        return 1.645
    return 1.96
