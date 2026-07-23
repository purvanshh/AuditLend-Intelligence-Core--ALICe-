"""Champion/challenger comparison with decision rules for rollback/promotion."""

from __future__ import annotations

from dataclasses import dataclass

from ml.causal.ab_framework import ExperimentResult


@dataclass
class ChampionChallengerDecision:
    winner: str
    profit_lift: float
    profit_ci: tuple[float, float]
    p_value: float
    significant: bool
    sample_size: int
    recommendation: str


def compare_arms(experiment_result: ExperimentResult) -> ChampionChallengerDecision:
    control_count = experiment_result.sample_size.get(experiment_result.config.control_arm, 0)
    treatment_count = experiment_result.sample_size.get(experiment_result.config.treatment_arm, 0)
    total_sample = control_count + treatment_count

    min_required = experiment_result.config.min_sample_size
    if control_count < min_required or treatment_count < min_required:
        return ChampionChallengerDecision(
            winner="tie",
            profit_lift=experiment_result.deltas.get("profit", 0.0),
            profit_ci=experiment_result.confidence_intervals.get("profit", (0.0, 0.0)),
            p_value=experiment_result.p_values.get("profit", 1.0),
            significant=False,
            sample_size=total_sample,
            recommendation=(
                f"CONTINUE: insufficient sample size "
                f"(control={control_count}, treatment={treatment_count}, "
                f"min_required={min_required})"
            ),
        )

    profit_delta = experiment_result.deltas.get("profit", 0.0)
    ci_lower, ci_upper = experiment_result.confidence_intervals.get("profit", (0.0, 0.0))
    p_value = experiment_result.p_values.get("profit", 1.0)
    significant = experiment_result.significant.get("profit", False)

    if significant and ci_lower > 0.0 and profit_delta > 0.0:
        return ChampionChallengerDecision(
            winner="challenger",
            profit_lift=profit_delta,
            profit_ci=(ci_lower, ci_upper),
            p_value=p_value,
            significant=True,
            sample_size=total_sample,
            recommendation=(
                f"PROMOTE: challenger shows significant profit lift of "
                f"{profit_delta:.4f} (CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
                f"p={p_value:.4f})"
            ),
        )

    if significant and ci_upper < 0.0 and profit_delta < 0.0:
        return ChampionChallengerDecision(
            winner="champion",
            profit_lift=profit_delta,
            profit_ci=(ci_lower, ci_upper),
            p_value=p_value,
            significant=True,
            sample_size=total_sample,
            recommendation=(
                f"ROLLBACK: challenger shows significant profit decrease of "
                f"{profit_delta:.4f} (CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
                f"p={p_value:.4f})"
            ),
        )

    return ChampionChallengerDecision(
        winner="tie",
        profit_lift=profit_delta,
        profit_ci=(ci_lower, ci_upper),
        p_value=p_value,
        significant=False,
        sample_size=total_sample,
        recommendation=(
            f"CONTINUE: no significant difference in profit between arms "
            f"(delta={profit_delta:.4f}, p={p_value:.4f}, "
            f"CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
        ),
    )
