"""Quantify temperature bias from fitting memory data with a Markov model."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.model import CollisionParameters, record_distribution


def main() -> None:
    output = ROOT / "results" / "misspecification"
    output.mkdir(parents=True, exist_ok=True)

    record_length = 8
    probe_angle = 0.5 * np.pi
    true_temperatures = np.linspace(0.40, 2.40, 9)
    memories = np.linspace(0.0, 1.30, 11)
    candidate_temperatures = np.linspace(0.20, 3.20, 181)

    # Markov likelihood family used by the misspecified estimator. These
    # distributions can be reused for every true (T, mu) point.
    markov_family = np.asarray(
        [
            record_distribution(
                CollisionParameters(float(temperature), 0.0),
                probe_angle,
                record_length,
            )
            for temperature in candidate_temperatures
        ]
    )
    log_markov = np.log(np.maximum(markov_family, 1e-300))

    fitted = np.zeros((true_temperatures.size, memories.size))
    bias = np.zeros_like(fitted)
    minimum_kl = np.zeros_like(fitted)

    rows: list[dict[str, float]] = []
    total = true_temperatures.size * memories.size
    completed = 0
    for i, true_temperature in enumerate(true_temperatures):
        for j, memory in enumerate(memories):
            true_distribution = record_distribution(
                CollisionParameters(float(true_temperature), float(memory)),
                probe_angle,
                record_length,
            )
            cross_entropy = -(log_markov @ true_distribution)
            best_index = int(np.argmin(cross_entropy))
            fitted_temperature = float(candidate_temperatures[best_index])
            entropy = -float(
                np.sum(
                    true_distribution
                    * np.log(np.maximum(true_distribution, 1e-300))
                )
            )
            kl = float(cross_entropy[best_index] - entropy)

            fitted[i, j] = fitted_temperature
            bias[i, j] = fitted_temperature - true_temperature
            minimum_kl[i, j] = max(kl, 0.0)
            rows.append(
                {
                    "true_temperature": float(true_temperature),
                    "memory_angle": float(memory),
                    "fitted_markov_temperature": fitted_temperature,
                    "bias": float(bias[i, j]),
                    "relative_bias": float(bias[i, j] / true_temperature),
                    "minimum_kl": float(minimum_kl[i, j]),
                }
            )
            completed += 1
            if completed % max(total // 10, 1) == 0:
                print(f"completed {completed}/{total}")

    with (output / "misspecification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        output / "misspecification_data.npz",
        true_temperatures=true_temperatures,
        memories=memories,
        candidate_temperatures=candidate_temperatures,
        fitted=fitted,
        bias=bias,
        minimum_kl=minimum_kl,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    limit = float(np.max(np.abs(bias)))
    image = ax.imshow(
        bias,
        origin="lower",
        aspect="auto",
        extent=[
            memories.min(),
            memories.max(),
            true_temperatures.min(),
            true_temperatures.max(),
        ],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_xlabel(r"true memory angle $\mu$")
    ax.set_ylabel(r"true temperature $T/\omega$")
    ax.set_title("Asymptotic bias from a Markovian likelihood")
    fig.colorbar(image, ax=ax, label=r"pseudo-true bias $T_* - T$")
    fig.savefig(output / "markov_fit_bias.png", dpi=180)
    plt.close(fig)

    relative_bias = bias / true_temperatures[:, None]
    nonzero_memory = memories > 0.0
    summary = {
        "record_length": record_length,
        "probe_angle": float(probe_angle),
        "max_absolute_bias": float(np.max(np.abs(bias[:, nonzero_memory]))),
        "max_absolute_relative_bias": float(
            np.max(np.abs(relative_bias[:, nonzero_memory]))
        ),
        "median_absolute_relative_bias": float(
            np.median(np.abs(relative_bias[:, nonzero_memory]))
        ),
        "fraction_over_10_percent_relative_bias": float(
            np.mean(np.abs(relative_bias[:, nonzero_memory]) >= 0.10)
        ),
        "max_minimum_kl": float(np.max(minimum_kl)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

