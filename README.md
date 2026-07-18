# tabflowmini

Fake tables are easy. Fake tables that don't quietly ruin your churn model are the actual assignment. tabflowmini grades itself on the second one.

![CI](https://github.com/ahmeddoghri/tabflowmini/actions/workflows/ci.yml/badge.svg)
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
