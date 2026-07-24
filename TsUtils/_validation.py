"""Shared private validation helpers for TsUtils."""

from __future__ import annotations

import numpy as np


def _resolve_missing_rows(finite_rows, missing, *, name="data"):
    """Validate a missing-value policy and return dropped row positions."""
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'")

    finite_rows = np.asarray(finite_rows, dtype=bool)
    if finite_rows.ndim != 1:
        raise ValueError("finite_rows must be one-dimensional")

    dropped_positions = tuple(
        int(position) for position in np.flatnonzero(~finite_rows)
    )
    if dropped_positions and missing == "raise":
        positions = ", ".join(str(position) for position in dropped_positions)
        raise ValueError(
            f"{name} contains non-finite values at row positions: {positions}"
        )
    return dropped_positions
