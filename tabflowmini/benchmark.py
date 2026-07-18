from __future__ import annotations

import json
from pathlib import Path

from .data import make_table
from .synth import evaluate_synth, fit_synth, sample_synth


def main() -> None:
    real = make_table()
    model = fit_synth(real)
    synth = sample_synth(model, len(real))
    metrics = evaluate_synth(real, synth)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("tabflowmini benchmark: tabular synthetic data")
    for key, value in metrics.items():
        print(f"{key:16s} {value:.3f}")
    print("artifact         artifacts/metrics.json")


if __name__ == "__main__":
    main()
