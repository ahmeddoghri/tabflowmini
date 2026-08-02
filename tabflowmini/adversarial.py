from __future__ import annotations

import math
import random

from .data import Row

# TUNING_SEEDS: used to characterize and tune the churn-label fix.
# HOLDOUT_SEEDS: disjoint, evaluated exactly once after the fix was finalized.
TUNING_SEEDS = list(range(1000, 1020))
HOLDOUT_SEEDS = list(range(2000, 2015))


def make_table_variant(n: int = 700, seed: int = 17) -> list[Row]:
    """Same structural form as data.make_table, but with a churn-label
    relationship whose coefficients are randomized per seed instead of
    matching synth.py's hardcoded formula. A synthesizer that actually
    learns the churn relationship from data should generalize here; one
    that hardcodes the original coefficients should not.
    """
    rng = random.Random(seed)
    coef_rng = random.Random(seed * 31 + 7)
    plans = ["basic", "pro", "enterprise"]

    w_visits = coef_rng.uniform(0.02, 0.30)
    w_income = coef_rng.uniform(-0.04, -0.002)
    intercept = coef_rng.uniform(-1.5, -0.2)
    plan_bonus = {
        "basic": coef_rng.uniform(0.05, 0.6),
        "pro": coef_rng.uniform(-0.3, 0.1),
        "enterprise": coef_rng.uniform(-0.7, -0.1),
    }

    rows = []
    for _ in range(n):
        age = max(18.0, min(82.0, rng.gauss(39, 11)))
        income = max(25.0, rng.gauss(72 + 0.8 * (age - 39), 20))
        visits = max(0.0, rng.gauss(6.5 - 0.025 * (income - 72), 2.0))
        plan = rng.choices(plans, weights=[0.52, 0.34, 0.14])[0]
        churn_p = 1.0 / (
            1.0 + math.exp(-(intercept + w_visits * visits + w_income * (income - 70) + plan_bonus[plan]))
        )
        churn = 1 if rng.random() < churn_p else 0
        rows.append(Row(age, income, visits, plan, churn))
    return rows
