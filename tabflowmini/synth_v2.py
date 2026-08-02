from __future__ import annotations

import math
from dataclasses import dataclass, field

from .data import Row
from .synth import SynthModel, fit_synth


@dataclass(frozen=True)
class ChurnWeights:
    visits: float
    income: float
    bias: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SynthModelV2:
    base: SynthModel
    churn_weights: ChurnWeights


def _fit_churn_logistic(rows: list[Row], plans: list[str], iters: int = 2000, lr: float = 0.3) -> ChurnWeights:
    """Fit a small logistic regression for churn from real rows instead of
    assuming the true generating coefficients are already known."""
    visits = [row.visits for row in rows]
    income = [row.income - 70 for row in rows]
    v_mean = sum(visits) / len(visits)
    v_std = math.sqrt(sum((x - v_mean) ** 2 for x in visits) / len(visits)) or 1.0
    i_mean = sum(income) / len(income)
    i_std = math.sqrt(sum((x - i_mean) ** 2 for x in income) / len(income)) or 1.0

    xv = [(x - v_mean) / v_std for x in visits]
    xi = [(x - i_mean) / i_std for x in income]

    w_v = 0.0
    w_i = 0.0
    bias = dict.fromkeys(plans, 0.0)
    n = len(rows)
    for _ in range(iters):
        g_v = g_i = 0.0
        g_bias = dict.fromkeys(plans, 0.0)
        for row, a, b in zip(rows, xv, xi):
            z = bias[row.plan] + w_v * a + w_i * b
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - row.churn
            g_v += err * a
            g_i += err * b
            g_bias[row.plan] += err
        w_v -= lr * g_v / n
        w_i -= lr * g_i / n
        for plan in plans:
            bias[plan] -= lr * g_bias[plan] / n

    raw_w_v = w_v / v_std
    raw_w_i = w_i / i_std
    raw_bias = {plan: bias[plan] - w_v * v_mean / v_std - w_i * i_mean / i_std for plan in plans}
    return ChurnWeights(visits=raw_w_v, income=raw_w_i, bias=raw_bias)


def fit_synth_v2(rows: list[Row]) -> SynthModelV2:
    base = fit_synth(rows)
    weights = _fit_churn_logistic(rows, list(base.plan_probs))
    return SynthModelV2(base=base, churn_weights=weights)


def sample_synth_v2(model: SynthModelV2, n: int, seed: int = 3):
    import random as _random

    from .synth import _sample_plan

    base = model.base
    weights = model.churn_weights
    rng = _random.Random(seed)
    rows = []
    for _ in range(n):
        z1, z2, z3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        age = base.means[0] + base.stds[0] * z1
        income = base.means[1] + base.stds[1] * (
            base.corr[1][0] * z1 + math.sqrt(max(0.1, 1 - base.corr[1][0] ** 2)) * z2
        )
        visits = max(0.0, base.means[2] + base.stds[2] * (0.35 * z1 - 0.35 * z2 + 0.75 * z3))
        plan = _sample_plan(rng, base.plan_probs)
        score = weights.bias[plan] + weights.visits * visits + weights.income * (income - 70)
        churn = 1 if rng.random() < 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score)))) else 0
        rows.append(Row(age, income, visits, plan, churn))
    return rows
