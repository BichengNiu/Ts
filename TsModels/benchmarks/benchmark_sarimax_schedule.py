"""Measure the serial and bounded-process AutoSARIMAX scheduling policy.

Run from the repository root:

    python TsModels/benchmarks/benchmark_sarimax_schedule.py

The benchmark is deliberately outside pytest: runtime and process startup
depend on the host, whereas correctness belongs in the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import pandas as pd

# Running this file directly places ``benchmarks/`` first on sys.path, which
# could otherwise import an older installed Ts distribution.  The benchmark
# must measure the checkout it accompanies.
_CHECKOUT_PARENT = Path(__file__).resolve().parents[3]
if str(_CHECKOUT_PARENT) not in sys.path:
    sys.path.insert(0, str(_CHECKOUT_PARENT))

from Ts.TsModels import AutoSARIMAX

_REPEATS = 5
_WORKER_REQUESTS = (1, 2, 3, 4, 6)


def _benchmark_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Build the fixed 120-observation, two-exogenous-variable workload."""
    rng = np.random.default_rng(20260829)
    exog = pd.DataFrame(
        {
            "x1": rng.normal(size=120),
            "x2": rng.normal(size=120),
        }
    )
    innovations = rng.normal(scale=0.5, size=120)
    values = np.empty(120)
    values[0] = innovations[0]
    for index in range(1, len(values)):
        values[index] = (
            0.45 * values[index - 1]
            + 0.7 * exog.iloc[index, 0]
            - 0.35 * exog.iloc[index, 1]
            + innovations[index]
        )
    return pd.Series(values, name="y"), exog


def _fit_once(n_jobs: int) -> tuple[float, dict[str, int | float | str]]:
    """Fit the fixed 32-candidate grid once and return elapsed time/audit."""
    series, exog = _benchmark_inputs()
    started = perf_counter()
    result = AutoSARIMAX(
        series,
        exog=exog,
        p=(0, 3),
        d=(0, 1),
        q=(0, 3),
        P=(0, 0),
        D=(0, 0),
        Q=(0, 0),
        n_jobs=n_jobs,
    ).fit()
    return perf_counter() - started, result.search_metadata


def main() -> None:
    """Warm each mode, execute five repeats, and print a JSON comparison."""
    records = []
    for n_jobs in _WORKER_REQUESTS:
        _fit_once(n_jobs)
        timings = []
        metadata = {}
        for _ in range(_REPEATS):
            elapsed, metadata = _fit_once(n_jobs)
            timings.append(elapsed)
        records.append(
            {
                "requested_n_jobs": n_jobs,
                "median_seconds": median(timings),
                "samples_seconds": timings,
                "schedule": metadata,
            }
        )

    serial_seconds = records[0]["median_seconds"]
    for record in records[1:]:
        record["speedup_vs_serial"] = serial_seconds / record["median_seconds"]
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
