from __future__ import annotations

import random

from ml.models.uplift_xgb import (
    UpliftModel,
    compute_qini_coefficient,
    uplift_decision_rule,
)


def _make_synthetic_training_data(
    n_treatment: int = 200,
    n_control: int = 200,
):
    """Generate deterministic synthetic data where treatment reduces default.

    Good candidates (low risk): feature_1 high, feature_2 low
    Bad candidates (high risk): feature_1 low, feature_2 high

    Treatment effect: good candidates default less when approved,
    bad candidates default at similar rates regardless.
    """
    treatment_features: list[dict[str, float]] = []
    treatment_outcomes: list[int] = []
    control_features: list[dict[str, float]] = []
    control_outcomes: list[int] = []

    for i in range(n_treatment):
        t = (i * 7 % 100) / 100.0
        feature_1 = 0.1 + t * 0.9
        feature_2 = 0.9 - t * 0.9
        row = {"feature_1": feature_1, "feature_2": feature_2}
        treatment_features.append(row)
        default = 1 if feature_1 < 0.75 else 0
        treatment_outcomes.append(default)

    for i in range(n_control):
        t = (i * 7 % 100) / 100.0
        feature_1 = 0.1 + t * 0.9
        feature_2 = 0.9 - t * 0.9
        row = {"feature_1": feature_1, "feature_2": feature_2}
        control_features.append(row)
        default = 1 if feature_1 < 0.95 else 0
        control_outcomes.append(default)

    return treatment_features, treatment_outcomes, control_features, control_outcomes


def _make_good_candidates(n: int = 50):
    """Applicants who should benefit from approval (low default risk)."""
    return [
        {"feature_1": 0.9 + (i % 10) * 0.01, "feature_2": 0.1 - (i % 10) * 0.005}
        for i in range(n)
    ]


def _make_bad_candidates(n: int = 50):
    """Applicants who would default regardless of approval."""
    return [
        {"feature_1": 0.1 + (i % 10) * 0.01, "feature_2": 0.9 - (i % 10) * 0.005}
        for i in range(n)
    ]


def _make_mixed_candidates(n: int = 100):
    """Mix of good and bad candidates."""
    good = _make_good_candidates(n // 2)
    bad = _make_bad_candidates(n // 2)
    return good + bad


class TestUpliftModelFit:
    def test_fit_trains_both_models(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        result = model.fit(t_feat, t_out, c_feat, c_out)

        assert len(result.uplift_scores) == len(t_feat)
        assert len(result.treatment_probabilities) == len(t_feat)
        assert len(result.control_probabilities) == len(t_feat)
        assert result.n_treatment == len(t_feat)
        assert result.n_control == len(c_feat)

    def test_fit_raises_on_empty_treatment(self):
        model = UpliftModel()
        try:
            model.fit([], [], [{"a": 1.0}], [0])
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_fit_raises_on_empty_control(self):
        model = UpliftModel()
        try:
            model.fit([{"a": 1.0}], [0], [], [])
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestUpliftModelPredict:
    def test_predict_uplift_returns_scores_in_range(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates()
        result = model.predict_uplift(candidates)

        assert len(result.uplift_scores) == len(candidates)
        for score in result.uplift_scores:
            assert -1.0 <= score <= 1.0, f"Uplift score {score} out of range [-1, 1]"

    def test_uplift_negative_for_good_candidates(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        good = _make_good_candidates()
        result = model.predict_uplift(good)

        assert len(result.uplift_scores) > 0
        avg_uplift = sum(result.uplift_scores) / len(result.uplift_scores)
        assert avg_uplift < 0, (
            f"Expected negative avg uplift for good candidates, got {avg_uplift:.4f}"
        )

    def test_predict_raises_before_fit(self):
        model = UpliftModel()
        try:
            model.predict_uplift([{"feature_1": 0.5}])
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass

    def test_predict_empty_input(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        result = model.predict_uplift([])
        assert result.uplift_scores == []
        assert result.treatment_probabilities == []
        assert result.control_probabilities == []


class TestSegmentByUplift:
    def test_creates_correct_number_of_segments(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates(100)
        result = model.predict_uplift(candidates)
        segments = model.segment_by_uplift(candidates, result.uplift_scores, n_segments=5)

        assert segments["n_segments"] == 5
        assert len(segments["segments"]) == 5

    def test_segments_are_sorted_by_uplift(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates(100)
        result = model.predict_uplift(candidates)
        segments = model.segment_by_uplift(candidates, result.uplift_scores, n_segments=5)

        avg_uplifts = [s["avg_uplift"] for s in segments["segments"]]
        for i in range(len(avg_uplifts) - 1):
            assert avg_uplifts[i] <= avg_uplifts[i + 1], (
                f"Segment {i} avg uplift {avg_uplifts[i]} > segment {i + 1} {avg_uplifts[i + 1]}"
            )

    def test_segments_contain_recommendations(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates(100)
        result = model.predict_uplift(candidates)
        segments = model.segment_by_uplift(candidates, result.uplift_scores, n_segments=5)

        for segment in segments["segments"]:
            assert "recommendation" in segment
            assert segment["recommendation"] in (
                "Strongly recommend approval",
                "Use standard risk assessment",
                "Recommend decline",
            )

    def test_segment_empty_input(self):
        model = UpliftModel()
        result = model.segment_by_uplift([], [])
        assert result["n_segments"] == 0
        assert result["segments"] == []


class TestQiniCoefficient:
    def test_perfect_model_sorts_positive(self):
        outcomes = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
        treatment = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        treat_event = [i for i, (t, o) in enumerate(zip(treatment, outcomes)) if t == 1 and o == 1]
        ctrl_non_event = [i for i, (t, o) in enumerate(zip(treatment, outcomes)) if t == 0 and o == 0]
        perfect_order: list[int] = []
        while treat_event or ctrl_non_event:
            if treat_event:
                perfect_order.append(treat_event.pop(0))
            if ctrl_non_event:
                perfect_order.append(ctrl_non_event.pop(0))
        remaining = [i for i in range(len(outcomes)) if i not in perfect_order]
        perfect_order.extend(remaining)
        scores = [1.0 - perfect_order.index(i) * 0.1 for i in range(len(outcomes))]
        qini = compute_qini_coefficient(scores, outcomes, treatment)
        assert qini > 0.0, f"Expected positive Qini for well-sorted model, got {qini:.4f}"

    def test_random_model_near_zero(self):
        rng = random.Random(42)
        n = 200
        treatment = [1] * 100 + [0] * 100
        outcomes = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
        rng.shuffle(treatment)
        rng = random.Random(99)
        scores = [rng.random() for _ in range(n)]
        qini = compute_qini_coefficient(scores, outcomes, treatment)
        assert abs(qini) < 0.3, f"Expected near-zero Qini for random model, got {qini:.4f}"

    def test_worst_model_negative(self):
        outcomes = [1, 1, 0, 0, 0, 1, 0]
        treatment = [1, 1, 1, 1, 0, 0, 0]
        treat_event = [i for i, (t, o) in enumerate(zip(treatment, outcomes)) if t == 1 and o == 1]
        ctrl_non_event = [i for i, (t, o) in enumerate(zip(treatment, outcomes)) if t == 0 and o == 0]
        perfect_order: list[int] = []
        while treat_event or ctrl_non_event:
            if treat_event:
                perfect_order.append(treat_event.pop(0))
            if ctrl_non_event:
                perfect_order.append(ctrl_non_event.pop(0))
        remaining = [i for i in range(len(outcomes)) if i not in perfect_order]
        perfect_order.extend(remaining)
        reversed_order = list(reversed(perfect_order))
        scores = [1.0 - reversed_order.index(i) * 0.1 for i in range(len(outcomes))]
        qini = compute_qini_coefficient(scores, outcomes, treatment)
        assert qini < 0.0, f"Expected negative Qini for worst model, got {qini:.4f}"

    def test_empty_input(self):
        qini = compute_qini_coefficient([], [], [])
        assert qini == 0.0

    def test_unequal_lengths_raises(self):
        try:
            compute_qini_coefficient([0.5, 0.5], [0, 0], [1])
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_no_treated_returns_zero(self):
        qini = compute_qini_coefficient([0.5, 0.5], [0, 0], [0, 0])
        assert qini == 0.0

    def test_no_control_returns_zero(self):
        qini = compute_qini_coefficient([0.5, 0.5], [0, 0], [1, 1])
        assert qini == 0.0

    def test_synthetic_data_qini_range(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates(200)
        predict_result = model.predict_uplift(candidates)

        treat_labels = [1] * (len(candidates) // 2) + [0] * (len(candidates) // 2)
        outcomes = predict_result.treatment_probabilities
        outcomes_binary = [1 if p > 0.5 else 0 for p in outcomes]

        qini = compute_qini_coefficient(
            predict_result.uplift_scores,
            outcomes_binary,
            treat_labels,
        )
        assert qini >= -1.0
        assert qini <= 1.0


class TestUpliftDecisionRule:
    def test_strongly_recommend_approval(self):
        assert uplift_decision_rule(-0.1) == "Strongly recommend approval"
        assert uplift_decision_rule(-0.06) == "Strongly recommend approval"
        assert uplift_decision_rule(-0.051) == "Strongly recommend approval"

    def test_neutral(self):
        assert uplift_decision_rule(-0.04) == "Use standard risk assessment"
        assert uplift_decision_rule(0.0) == "Use standard risk assessment"
        assert uplift_decision_rule(0.04) == "Use standard risk assessment"
        assert uplift_decision_rule(0.05, threshold=0.0) == "Use standard risk assessment"

    def test_recommend_decline(self):
        assert uplift_decision_rule(0.06) == "Recommend decline"
        assert uplift_decision_rule(0.1) == "Recommend decline"
        assert uplift_decision_rule(1.0) == "Recommend decline"

    def test_boundary_negative(self):
        result = uplift_decision_rule(-0.0501)
        assert result == "Strongly recommend approval"

    def test_boundary_positive(self):
        result = uplift_decision_rule(0.0501)
        assert result == "Recommend decline"

    def test_boundary_neutral_lower(self):
        assert uplift_decision_rule(-0.0499) == "Use standard risk assessment"

    def test_boundary_neutral_upper(self):
        assert uplift_decision_rule(0.0499) == "Use standard risk assessment"


class TestFeatureImportance:
    def test_feature_importance_extracted(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        result = model.fit(t_feat, t_out, c_feat, c_out)

        assert len(result.feature_importance) > 0
        for key, value in result.feature_importance.items():
            assert value >= 0.0

    def test_feature_importance_in_predict(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates()
        result = model.predict_uplift(candidates)
        assert isinstance(result.feature_importance, dict)


class TestSyntheticIntegration:
    def test_end_to_end_with_synthetic_data(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data(
            n_treatment=500, n_control=500
        )
        model = UpliftModel()
        result = model.fit(t_feat, t_out, c_feat, c_out)

        assert result.n_treatment == 500
        assert result.n_control == 500

        candidates = _make_good_candidates(50) + _make_bad_candidates(50)
        predict_result = model.predict_uplift(candidates)
        assert len(predict_result.uplift_scores) == 100

        segments = model.segment_by_uplift(candidates, predict_result.uplift_scores, n_segments=5)
        assert segments["n_segments"] == 5

        good_scores = predict_result.uplift_scores[:50]
        bad_scores = predict_result.uplift_scores[50:]
        avg_good = sum(good_scores) / len(good_scores)
        avg_bad = sum(bad_scores) / len(bad_scores)
        assert avg_good < avg_bad, (
            f"Expected good candidates to have lower uplift, got good={avg_good:.4f} bad={avg_bad:.4f}"
        )

    def test_different_segment_counts(self):
        t_feat, t_out, c_feat, c_out = _make_synthetic_training_data()
        model = UpliftModel()
        model.fit(t_feat, t_out, c_feat, c_out)

        candidates = _make_mixed_candidates(100)
        result = model.predict_uplift(candidates)

        for n in [2, 3, 10]:
            segments = model.segment_by_uplift(candidates, result.uplift_scores, n_segments=n)
            assert segments["n_segments"] == n
            assert len(segments["segments"]) == n
