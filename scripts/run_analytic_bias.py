"""Validate the exact pseudo-true Markov temperature against the grid fit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.classical_full_swap import pseudo_true_markov_temperature
from memory_thermometry.model import CollisionParameters


def main() -> None:
    source_path = ROOT / "results" / "misspecification" / "misspecification_data.npz"
    if not source_path.exists():
        raise FileNotFoundError("run scripts/run_misspecification.py first")
    source = np.load(source_path)
    temperatures = source["true_temperatures"]
    memories = source["memories"]
    numerical_fitted = source["fitted"]
    record_length = 8

    analytic_finite = np.zeros_like(numerical_fitted)
    analytic_infinite = np.zeros_like(numerical_fitted)
    for i, temperature in enumerate(temperatures):
        for j, memory in enumerate(memories):
            parameters = CollisionParameters(float(temperature), float(memory))
            analytic_finite[i, j] = pseudo_true_markov_temperature(
                parameters, length=record_length
            )
            analytic_infinite[i, j] = pseudo_true_markov_temperature(parameters)

    finite_error = numerical_fitted - analytic_finite
    infinite_bias = analytic_infinite - temperatures[:, None]
    nonzero_memory = memories > 0.0

    output = ROOT / "results" / "analytic_bias"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "analytic_bias_data.npz",
        temperatures=temperatures,
        memories=memories,
        numerical_fitted=numerical_fitted,
        analytic_finite=analytic_finite,
        analytic_infinite=analytic_infinite,
        finite_error=finite_error,
        infinite_bias=infinite_bias,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    limit = float(np.max(np.abs(infinite_bias)))
    image = ax.imshow(
        infinite_bias,
        origin="lower",
        aspect="auto",
        extent=[memories.min(), memories.max(), temperatures.min(), temperatures.max()],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_xlabel(r"memory angle $\mu$")
    ax.set_ylabel(r"true temperature $T/\omega$")
    ax.set_title("Exact long-record Markov misspecification bias")
    fig.colorbar(image, ax=ax, label=r"$T_* - T$")
    fig.savefig(output / "exact_long_record_bias.png", dpi=180)
    plt.close(fig)

    summary = {
        "finite_block_length": record_length,
        "candidate_temperature_grid_spacing": float(
            source["candidate_temperatures"][1]
            - source["candidate_temperatures"][0]
        ),
        "max_grid_fit_minus_exact_finite_block": float(np.max(np.abs(finite_error))),
        "all_nonzero_memory_long_biases_negative": bool(
            np.all(infinite_bias[:, nonzero_memory] < 0.0)
        ),
        "median_absolute_long_relative_bias": float(
            np.median(
                np.abs(
                    infinite_bias[:, nonzero_memory]
                    / temperatures[:, None]
                )
            )
        ),
        "max_absolute_long_relative_bias": float(
            np.max(
                np.abs(
                    infinite_bias[:, nonzero_memory]
                    / temperatures[:, None]
                )
            )
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

