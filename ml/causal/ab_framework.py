"""Statistical experiment framework for champion/challenger A/B testing."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import scipy.stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


PROFIT_PER_LOAN = 0.12
LOSS_GIVEN_DEFAULT = 0.65
AB_BUCKET_COUNT = 10_000


@dataclass
class ExperimentConfig:
    name: str
    control_arm: str = "heuristic"
    treatment_arm: str = "ml"
    metrics: tuple[str, ...] = ("profit", "default_rate", "approval_rate", "calibration")
    alpha: float = 0.05
    min_sample_size: int = 1000
    stratified: bool = True


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    control_results: dict[str, float]
    treatment_results: dict[str, float]
    deltas: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    p_values: dict[str, float]
    significant: dict[str, bool]
    sample_size: dict[str, int]


class ExperimentFramework:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def assign(self, application_id: str, grade: str, purpose: str) -> str:
        if not self.config.stratified:
            digest = hashlib.sha256(application_id.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % AB_BUCKET_COUNT
        else:
            seed = f"{application_id}|{grade}|{purpose}"
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % AB_BUCKET_COUNT
        threshold = AB_BUCKET_COUNT // 2
        return self.config.control_arm if bucket < threshold else self.config.treatment_arm

    def analyze(self, records: list[dict[str, Any]]) -> ExperimentResult:
        control_key = self.config.control_arm
        treatment_key = self.config.treatment_arm

        control_records = [r for r in records if r.get("arm") == control_key]
        treatment_records = [r for r in records if r.get("arm") == treatment_key]

        control_results = self._compute_metrics(control_records)
        treatment_results = self._compute_metrics(treatment_records)

        deltas = {}
        all_metrics = set(control_results.keys()) | set(treatment_results.keys())
        for m in all_metrics:
            deltas[m] = round(treatment_results.get(m, 0.0) - control_results.get(m, 0.0), 6)

        ctrl_profit_values = self._extract_metric_values(control_records, "profit")
        trt_profit_values = self._extract_metric_values(treatment_records, "profit")

        profit_ci = self.bootstrap_ci(ctrl_profit_values, trt_profit_values, metric="profit")
        t_stat, p_value = self.welch_ttest(ctrl_profit_values, trt_profit_values)

        confidence_intervals: dict[str, tuple[float, float]] = {"profit": profit_ci}
        p_values: dict[str, float] = {"profit": p_value}
        significant: dict[str, bool] = {"profit": p_value < self.config.alpha}

        sample_size = {
            control_key: len(control_records),
            treatment_key: len(treatment_records),
        }

        return ExperimentResult(
            config=self.config,
            control_results=control_results,
            treatment_results=treatment_results,
            deltas=deltas,
            confidence_intervals=confidence_intervals,
            p_values=p_values,
            significant=significant,
            sample_size=sample_size,
        )

    def bootstrap_ci(
        self,
        control_values: list[float],
        treatment_values: list[float],
        metric: str = "profit",
        n_bootstrap: int = 10_000,
    ) -> tuple[float, float]:
        rng = random.Random(42)
        ctrl = list(control_values)
        trt = list(treatment_values)
        if not ctrl or not trt:
            return (0.0, 0.0)
        n = len(ctrl)
        m = len(trt)

        def stat(samples: list[float]) -> float:
            return sum(samples) / len(samples)

        deltas: list[float] = []
        for _ in range(n_bootstrap):
            c_sample = [rng.choice(ctrl) for _ in range(n)]
            t_sample = [rng.choice(trt) for _ in range(m)]
            delta = stat(t_sample) - stat(c_sample)
            deltas.append(delta)
        deltas.sort()
        lower = deltas[int(0.025 * n_bootstrap)]
        upper = deltas[int(0.975 * n_bootstrap)]
        return (round(lower, 6), round(upper, 6))

    def welch_ttest(
        self, control_values: list[float], treatment_values: list[float]
    ) -> tuple[float, float]:
        if not control_values or not treatment_values:
            return (0.0, 1.0)
        n1 = len(control_values)
        n2 = len(treatment_values)
        mean1 = sum(control_values) / n1
        mean2 = sum(treatment_values) / n2
        var1 = sum((x - mean1) ** 2 for x in control_values) / (n1 - 1) if n1 > 1 else 0.0
        var2 = sum((x - mean2) ** 2 for x in treatment_values) / (n2 - 1) if n2 > 1 else 0.0

        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0.0:
            return (0.0, 1.0)

        t_stat = (mean2 - mean1) / se
        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        df = df_num / df_den if df_den > 0 else min(n1, n2) - 1

        if _HAS_SCIPY:
            p_value = float(_scipy_stats.t.sf(abs(t_stat), df) * 2.0)
        else:
            p_value = self._approximate_t_pvalue(abs(t_stat), df)
        return (round(t_stat, 6), round(p_value, 6))

    def _compute_metrics(self, records: list[dict[str, Any]]) -> dict[str, float]:
        if not records:
            return {"profit": 0.0, "default_rate": 0.0, "approval_rate": 0.0, "calibration": 0.0}

        n = len(records)
        approved = [r for r in records if r.get("decision") == "APPROVE"]
        defaults = [r for r in approved if int(r.get("defaulted", 0)) == 1]

        approval_rate = len(approved) / n if n else 0.0
        default_rate = len(defaults) / len(approved) if approved else 0.0

        total_profit = sum(
            -float(r["loan_amount"]) * LOSS_GIVEN_DEFAULT
            if int(r.get("defaulted", 0)) == 1
            else float(r["loan_amount"]) * PROFIT_PER_LOAN
            for r in approved
        )
        profit_per_app = total_profit / n if n else 0.0

        calibration = sum(float(r.get("confidence", 0.0)) for r in records) / n if n else 0.0

        return {
            "profit": round(profit_per_app, 6),
            "default_rate": round(default_rate, 6),
            "approval_rate": round(approval_rate, 6),
            "calibration": round(calibration, 6),
        }

    def _extract_metric_values(
        self, records: list[dict[str, Any]], metric: str
    ) -> list[float]:
        if metric == "profit":
            return [
                -float(r["loan_amount"]) * LOSS_GIVEN_DEFAULT
                if int(r.get("defaulted", 0)) == 1 and r.get("decision") == "APPROVE"
                else float(r["loan_amount"]) * PROFIT_PER_LOAN
                if r.get("decision") == "APPROVE"
                else 0.0
                for r in records
            ]
        if metric == "default_rate":
            return [float(r.get("defaulted", 0)) for r in records if r.get("decision") == "APPROVE"]
        if metric == "calibration":
            return [float(r.get("confidence", 0.0)) for r in records]
        return [float(r.get(metric, 0.0)) for r in records]

    def _approximate_t_pvalue(self, t_abs: float, df: float) -> float:
        if df <= 0:
            return 1.0
        if df > 100:
            return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(t_abs / math.sqrt(2.0))))
        x = df / (df + t_abs * t_abs)
        return 2.0 * (1.0 - 0.5 * self._reg_inc_beta(df / 2.0, 0.5, x))

    @staticmethod
    def _reg_inc_beta(a: float, b: float, x: float) -> float:
        if x < 0.0 or x > 1.0:
            return 0.0
        if x == 0.0 or x == 1.0:
            return float(x)

        if x > (a + 1.0) / (a + b + 2.0):
            return 1.0 - ExperimentFramework._reg_inc_beta(b, a, 1.0 - x)

        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta - math.log(a))

        f = 1.0
        c = 1.0
        d = 1.0 - (a + b) * x / (a + 1.0)
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        f = d

        for m in range(1, 201):
            numer_even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
            d = 1.0 + numer_even * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + numer_even / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            f *= d * c

            numer_odd = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
            d = 1.0 + numer_odd * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + numer_odd / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            f *= delta

            if abs(delta - 1.0) < 1e-10:
                break

        return front * f
