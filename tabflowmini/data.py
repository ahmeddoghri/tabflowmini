from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    age: float
    income: float
    visits: float
    plan: str
    churn: int


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_table(n: int = 700, seed: int = 17) -> list[Row]:
    rng = random.Random(seed)
    plans = ["basic", "pro", "enterprise"]
    rows = []
    for _ in range(n):
        age = max(18.0, min(82.0, rng.gauss(39, 11)))
        income = max(25.0, rng.gauss(72 + 0.8 * (age - 39), 20))
        visits = max(0.0, rng.gauss(6.5 - 0.025 * (income - 72), 2.0))
        plan = rng.choices(plans, weights=[0.52, 0.34, 0.14])[0]
        plan_bonus = {"basic": 0.45, "pro": -0.1, "enterprise": -0.55}[plan]
        churn_p = _sigmoid(-1.0 + 0.08 * visits - 0.015 * (income - 70) + plan_bonus)
        churn = 1 if rng.random() < churn_p else 0
        rows.append(Row(age, income, visits, plan, churn))
    return rows
