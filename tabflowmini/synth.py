from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from .data import Row


@dataclass(frozen=True)
class SynthModel:
    means: list[float]
    stds: list[float]
    corr: list[list[float]]
    plan_probs: dict[str, float]
    churn_rate: float


def _cols(rows: list[Row]) -> list[list[float]]:
    return [[row.age for row in rows], [row.income for row in rows], [row.visits for row in rows]]


def fit_synth(rows: list[Row]) -> SynthModel:
    cols = _cols(rows)
    means = [sum(col) / len(col) for col in cols]
    stds = [math.sqrt(sum((v - m) ** 2 for v in col) / len(col)) for col, m in zip(cols, means)]
    corr = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            cov = sum((cols[i][k] - means[i]) * (cols[j][k] - means[j]) for k in range(len(rows))) / len(rows)
            corr[i][j] = cov / max(1e-9, stds[i] * stds[j])
    plans = Counter(row.plan for row in rows)
    plan_probs = {plan: count / len(rows) for plan, count in plans.items()}
    churn_rate = sum(row.churn for row in rows) / len(rows)
    return SynthModel(means, stds, corr, plan_probs, churn_rate)


def _sample_plan(rng: random.Random, probs: dict[str, float]) -> str:
    pick = rng.random()
    acc = 0.0
    for plan, prob in probs.items():
        acc += prob
        if acc >= pick:
            return plan
    return next(iter(probs))


def sample_synth(model: SynthModel, n: int, seed: int = 3) -> list[Row]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z1, z2, z3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        age = model.means[0] + model.stds[0] * z1
        income = model.means[1] + model.stds[1] * (model.corr[1][0] * z1 + math.sqrt(max(0.1, 1 - model.corr[1][0] ** 2)) * z2)
        visits = model.means[2] + model.stds[2] * (0.35 * z1 - 0.35 * z2 + 0.75 * z3)
        plan = _sample_plan(rng, model.plan_probs)
        churn_score = -1.0 + 0.08 * visits - 0.015 * (income - 70) + {"basic": 0.45, "pro": -0.1, "enterprise": -0.55}[plan]
        churn = 1 if rng.random() < 1.0 / (1.0 + math.exp(-churn_score)) else 0
        rows.append(Row(age, income, max(0.0, visits), plan, churn))
    return rows


def _ks(a: list[float], b: list[float]) -> float:
    values = sorted(set(a + b))
    best = 0.0
    for value in values:
        ca = sum(1 for x in a if x <= value) / len(a)
        cb = sum(1 for x in b if x <= value) / len(b)
        best = max(best, abs(ca - cb))
    return best


def _nearest_duplicate_rate(real: list[Row], synth: list[Row]) -> float:
    real_keys = {(round(row.age, 1), round(row.income, 1), round(row.visits, 1), row.plan, row.churn) for row in real}
    synth_keys = {(round(row.age, 1), round(row.income, 1), round(row.visits, 1), row.plan, row.churn) for row in synth}
    return len(real_keys & synth_keys) / max(1, len(synth_keys))


def evaluate_synth(real: list[Row], synth: list[Row]) -> dict[str, float]:
    real_cols = _cols(real)
    synth_cols = _cols(synth)
    ks = sum(_ks(a, b) for a, b in zip(real_cols, synth_cols)) / 3.0
    plan_gap = sum(abs(Counter(row.plan for row in real)[plan] / len(real) - Counter(row.plan for row in synth)[plan] / len(synth)) for plan in {"basic", "pro", "enterprise"})
    churn_gap = abs(sum(row.churn for row in real) / len(real) - sum(row.churn for row in synth) / len(synth))
    duplicate_rate = _nearest_duplicate_rate(real, synth)
    return {
        "mean_ks": ks,
        "plan_gap": plan_gap,
        "churn_gap": churn_gap,
        "duplicate_rate": duplicate_rate,
    }
