from tabflowmini import evaluate_synth, fit_synth, make_table, sample_synth


def test_synth_generates_requested_size() -> None:
    model = fit_synth(make_table(n=100))
    assert len(sample_synth(model, 37)) == 37


def test_synth_preserves_marginals() -> None:
    real = make_table()
    synth = sample_synth(fit_synth(real), len(real))
    metrics = evaluate_synth(real, synth)
    assert metrics["mean_ks"] < 0.10
    assert metrics["plan_gap"] < 0.12


def test_duplicate_rate_is_low() -> None:
    real = make_table()
    synth = sample_synth(fit_synth(real), len(real))
    assert evaluate_synth(real, synth)["duplicate_rate"] < 0.05
