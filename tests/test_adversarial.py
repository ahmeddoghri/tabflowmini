from __future__ import annotations

from tabflowmini.adversarial import HOLDOUT_SEEDS, TUNING_SEEDS, make_table_variant
from tabflowmini.data import make_table
from tabflowmini.eval_v2 import _churn_gap, _summarize
from tabflowmini.synth import evaluate_synth, fit_synth, sample_synth
from tabflowmini.synth_v2 import fit_synth_v2, sample_synth_v2


def test_holdout_disjoint_from_tuning() -> None:
    assert set(TUNING_SEEDS).isdisjoint(HOLDOUT_SEEDS)


def test_original_benchmark_still_reproduces_exactly() -> None:
    real = make_table()
    model = fit_synth(real)
    synth = sample_synth(model, len(real))
    metrics = evaluate_synth(real, synth)
    assert round(metrics["mean_ks"], 3) == 0.056
    assert round(metrics["plan_gap"], 3) == 0.040
    assert round(metrics["churn_gap"], 3) == 0.026
    assert round(metrics["duplicate_rate"], 3) == 0.000


def test_original_bug_hardcoded_coefficients_only_work_on_the_original_dgp() -> None:
    """synth.sample_synth's churn formula is the exact coefficients data.make_table
    uses to generate churn, not something derived from the fitted model. On a table
    with a different (but equally plausible) true churn relationship, the churn_gap
    should degrade sharply, proving the low published churn_gap is not evidence of
    a learned label relationship."""
    real = make_table_variant(seed=1000)
    model = fit_synth(real)
    synth = sample_synth(model, len(real), seed=1000)
    gap = _churn_gap(real, synth)
    assert gap > 0.05


def test_churn_rate_field_is_unused_by_original_sampler() -> None:
    real = make_table()
    model = fit_synth(real)
    assert model.churn_rate == sum(row.churn for row in real) / len(real)
    # sample_synth never references model.churn_rate; two models with the same
    # stats but different churn_rate produce identical synthetic churn draws.
    from dataclasses import replace

    tampered = replace(model, churn_rate=0.0)
    synth_a = sample_synth(model, len(real), seed=3)
    synth_b = sample_synth(tampered, len(real), seed=3)
    assert [row.churn for row in synth_a] == [row.churn for row in synth_b]


def test_v2_fix_generalizes_on_tuning_seeds() -> None:
    result = _summarize(TUNING_SEEDS)
    assert result["learned_mean_churn_gap"] < result["hardcoded_mean_churn_gap"]
    assert result["learned_mean_churn_gap"] < 0.05


def test_v2_fix_generalizes_on_frozen_holdout_seeds() -> None:
    result = _summarize(HOLDOUT_SEEDS)
    assert result["learned_mean_churn_gap"] < result["hardcoded_mean_churn_gap"]
    assert result["learned_mean_churn_gap"] < 0.05


def test_v2_does_not_regress_the_original_published_table() -> None:
    real = make_table()
    v2_model = fit_synth_v2(real)
    v2_synth = sample_synth_v2(v2_model, len(real), seed=3)
    gap = _churn_gap(real, v2_synth)
    assert gap < 0.05


def test_v2_preserves_marginal_and_duplicate_metrics() -> None:
    real = make_table()
    v2_model = fit_synth_v2(real)
    v2_synth = sample_synth_v2(v2_model, len(real), seed=3)
    metrics = evaluate_synth(real, v2_synth)
    assert metrics["mean_ks"] < 0.10
    assert metrics["duplicate_rate"] < 0.05


def test_original_data_and_synth_modules_are_untouched() -> None:
    import inspect

    from tabflowmini import synth

    source = inspect.getsource(synth.sample_synth)
    assert "0.08 * visits" in source
    assert "0.015 * (income - 70)" in source


def test_report_is_reproducible() -> None:
    result_a = _summarize(TUNING_SEEDS[:5])
    result_b = _summarize(TUNING_SEEDS[:5])
    assert result_a == result_b
