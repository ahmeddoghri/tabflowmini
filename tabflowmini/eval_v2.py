from __future__ import annotations

from .adversarial import HOLDOUT_SEEDS, TUNING_SEEDS, make_table_variant
from .synth import fit_synth, sample_synth
from .synth_v2 import fit_synth_v2, sample_synth_v2


def _churn_gap(real, synth) -> float:
    return abs(sum(r.churn for r in real) / len(real) - sum(r.churn for r in synth) / len(synth))


def _summarize(seeds: list[int]) -> dict[str, float]:
    hardcoded_gaps = []
    learned_gaps = []
    for seed in seeds:
        real = make_table_variant(seed=seed)
        base_model = fit_synth(real)
        hardcoded_synth = sample_synth(base_model, len(real), seed=seed)
        hardcoded_gaps.append(_churn_gap(real, hardcoded_synth))

        v2_model = fit_synth_v2(real)
        learned_synth = sample_synth_v2(v2_model, len(real), seed=seed)
        learned_gaps.append(_churn_gap(real, learned_synth))

    return {
        "n": len(seeds),
        "hardcoded_mean_churn_gap": sum(hardcoded_gaps) / len(seeds),
        "learned_mean_churn_gap": sum(learned_gaps) / len(seeds),
    }


def main() -> None:
    real = make_table_variant(seed=17)
    original_model = fit_synth(real)
    original_synth = sample_synth(original_model, len(real), seed=3)
    original_gap = _churn_gap(real, original_synth)
    print("tabflowmini eval_v2: churn-label leakage check")
    print("published-benchmark-shaped table, but with a different true churn relationship:")
    print(f"  hardcoded-coefficient churn_gap  {original_gap:.3f}")

    tuning = _summarize(TUNING_SEEDS)
    holdout = _summarize(HOLDOUT_SEEDS)
    print()
    print("seeds        n    hardcoded churn_gap   learned churn_gap")
    print(f"tuning     {tuning['n']:3d}    {tuning['hardcoded_mean_churn_gap']:19.3f}   {tuning['learned_mean_churn_gap']:17.3f}")
    print(f"holdout    {holdout['n']:3d}    {holdout['hardcoded_mean_churn_gap']:19.3f}   {holdout['learned_mean_churn_gap']:17.3f}")


if __name__ == "__main__":
    main()
