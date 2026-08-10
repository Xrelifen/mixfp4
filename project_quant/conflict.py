"""N/K conflict decomposition utilities."""

from __future__ import annotations

from typing import Any


def decompose_nk_conflict(
    oracle: dict[str, Any],
    k_only: dict[str, Any],
    n_only: dict[str, Any],
    nk: dict[str, Any],
) -> dict[str, float]:
    base = float(oracle["oracle_error"])
    k_regret = float(k_only["constrained_error"]) - base
    n_regret = float(n_only["constrained_error"]) - base
    nk_regret = float(nk["constrained_error"]) - base
    return {
        "regret_oracle_to_k64": k_regret,
        "regret_oracle_to_n8k16": n_regret,
        "regret_oracle_to_n8k64": nk_regret,
        "nk_interaction_residual": nk_regret - k_regret - n_regret,
    }
