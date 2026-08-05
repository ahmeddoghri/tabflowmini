# tabflowmini

Fake tables are easy. Fake tables that don't quietly ruin your churn model are the actual assignment. tabflowmini grades itself on the second one.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Synthetic tabular data can look completely plausible and still be worthless.
It can match the column means and ruin the category balance, or preserve the
categories and quietly leak entire rows, or worst of all, break the exact
label relationship that made the table valuable in the first place.
tabflowmini fits a compact Gaussian transport model over continuous columns
(the readable stand-in for rectified flow, no hidden neural net required),
samples categorical marginals separately, regenerates the churn label from
the synthetic features, and then actually checks all of that instead of
declaring victory after one histogram looks close enough.

## Run it

```bash
git clone https://github.com/ahmeddoghri/tabflowmini
cd tabflowmini
pip install -e ".[dev]"
python -m tabflowmini.benchmark
```

## Verified benchmark

Generated locally with `python -m tabflowmini.benchmark`:

```text
mean_ks          0.056
plan_gap         0.040
churn_gap        0.026
duplicate_rate   0.000
```

Mean KS distance of 0.056 across continuous columns, a 0.040 gap on plan
distribution, a 0.026 gap on the churn rate the whole exercise exists to
protect, and zero duplicate rows leaking straight out of the training set.
This is a small audit harness, not a magic anonymizer, and it will tell you
exactly which of those four numbers to worry about before you ship the
synthetic table anywhere near production.

**Update:** that 0.026 churn_gap isn't evidence the model learned the churn
relationship. `sample_synth`'s churn formula is the literal coefficients
`data.make_table` uses to generate churn (`-1.0 + 0.08 * visits - 0.015 *
(income - 70) + plan_bonus`) — copy-pasted, not fitted. `fit_synth` even
computes a `churn_rate` field that the sampler never reads. On a table with
a different (but equally plausible) true churn relationship, this same
sampler's churn_gap blows up to 0.17-0.21. `sample_synth_v2` fits a real
logistic regression on the observed columns instead, and generalizes to
both the original table and randomized alternate relationships. Original
numbers above are untouched. Details below.

## The churn label wasn't learned, it was copied

`synth.sample_synth` computes churn with
`-1.0 + 0.08 * visits - 0.015 * (income - 70) + plan_bonus`, the exact
same formula `data.make_table` uses to generate the "real" table in the
first place. It isn't derived from `fit_synth`'s fitted means, stds, or
correlations — it's the answer, hardcoded. `SynthModel.churn_rate` is
computed from real data and then never referenced by the sampler at all.

```bash
python -m tabflowmini.eval_v2
```
```
seeds        n    hardcoded churn_gap   learned churn_gap
tuning      20                  0.173               0.012
holdout     15                  0.206               0.010
```

`adversarial.make_table_variant` builds tables structurally identical to
`data.make_table` but with randomized true churn coefficients per seed.
The unmodified `sample_synth` — which only "knows" the original table's
coefficients — degrades to a mean churn_gap of 0.17-0.21 on these tables.
`synth_v2.sample_synth_v2` fits an actual (feature-standardized) logistic
regression against each real table's observed visits/income/plan columns
and holds a churn_gap near 0.01 on both the tuning sweep and a disjoint
15-seed holdout evaluated once. `data.py`/`synth.py` are untouched, and
the published numbers above still reproduce exactly; `sample_synth_v2` is
opt-in.

## Research trail

- Flow matching for generative modeling, 2024 tutorial: https://arxiv.org/abs/2412.06264
- Diffusion models for tabular data, 2024: https://arxiv.org/abs/2407.02549
- TabDiff, 2024: https://arxiv.org/html/2410.20626v1
- Flow matching for tabular data synthesis, 2025: https://arxiv.org/abs/2512.00698
- SynthEval utility and privacy evaluation, 2024: https://arxiv.org/abs/2404.15821

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
