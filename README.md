# tabflowmini

A dependency-light tabular synthetic data demo inspired by rectified flow and
Gaussian transport. It fits a compact transport model over continuous columns,
samples categorical marginals, regenerates the label, and reports utility and
privacy-adjacent checks.

![CI](https://github.com/ahmeddoghri/tabflowmini/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

## Run it

```bash
git clone https://github.com/ahmeddoghri/tabflowmini
cd tabflowmini
pip install -e ".[dev]"
python -m tabflowmini.benchmark
```

## Verified benchmark

These numbers were generated locally with `python -m tabflowmini.benchmark`:

```text
mean_ks          0.056
plan_gap         0.040
churn_gap        0.026
duplicate_rate   0.000
```

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
