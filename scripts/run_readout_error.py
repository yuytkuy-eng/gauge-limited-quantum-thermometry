"""Map competition between environmental memory and unmodeled readout errors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.classical_full_swap import (
    critical_ignored_readout_error,
    pseudo_true_markov_temperature,
    pseudo_true_temperature_with_readout_error,
)
from memory_thermometry.model import CollisionParameters


def main() -> None:
    output = ROOT / "results" / "readout_error"
    output.mkdir(parents=True, exist_ok=True)

    temperatures = np.linspace(0.40, 2.40, 41)
    memories = np.linspace(0.0, 1.30, 53)
    critical = np.zeros((temperatures.size, memories.size))
    calibrated_difference = np.zeros_like(critical)

    for i, temperature in enumerate(temperatures):
        for j, memory in enumerate(memories):
            parameters = CollisionParameters(float(temperature), float(memory))
            critical[i, j] = critical_ignored_readout_error(parameters)
            ideal = pseudo_true_markov_temperature(parameters)
            calibrated = pseudo_true_temperature_with_readout_error(
                parameters,
                true_readout_error=0.03,
                assumed_readout_error=0.03,
            )
            calibrated_difference[i, j] = calibrated - ideal

    representative_temperature = 0.90
    error_rates = np.linspace(0.0, 0.03, 61)
    bias = np.zeros((error_rates.size, memories.size))
    for i, error in enumerate(error_rates):
        for j, memory in enumerate(memories):
            parameters = CollisionParameters(
                representative_temperature, float(memory)
            )
            fitted = pseudo_true_temperature_with_readout_error(
                parameters,
                true_readout_error=float(error),
                assumed_readout_error=0.0,
            )
            bias[i, j] = fitted - representative_temperature

    np.savez(
        output / "readout_error_data.npz",
        temperatures=temperatures,
        memories=memories,
        critical=critical,
        calibrated_difference=calibrated_difference,
        representative_temperature=representative_temperature,
        error_rates=error_rates,
        bias=bias,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    image = ax.imshow(
        100.0 * critical,
        origin="lower",
        aspect="auto",
        extent=[memories.min(), memories.max(), temperatures.min(), temperatures.max()],
    )
    ax.set_xlabel(r"memory angle $\mu$")
    ax.set_ylabel(r"temperature $T/\omega$")
    ax.set_title("Unmodeled readout error that cancels memory bias")
    fig.colorbar(image, ax=ax, label="critical bit-flip error (%)")
    fig.savefig(output / "critical_readout_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    finite_bias = bias[np.isfinite(bias)]
    limit = float(np.max(np.abs(finite_bias)))
    image = ax.imshow(
        bias,
        origin="lower",
        aspect="auto",
        extent=[memories.min(), memories.max(), 100 * error_rates.min(), 100 * error_rates.max()],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ax.contour(
        memories,
        100.0 * error_rates,
        bias,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.set_xlabel(r"memory angle $\mu$")
    ax.set_ylabel("unmodeled bit-flip error (%)")
    ax.set_title(r"Bias sign competition at $T/\omega=0.9$")
    fig.colorbar(image, ax=ax, label=r"$T_* - T$")
    fig.savefig(output / "bias_sign_competition.png", dpi=180)
    plt.close(fig)

    nonzero_memory = memories > 0.0
    representative = CollisionParameters(
        representative_temperature, memory_angle=0.50
    )
    representative_critical = critical_ignored_readout_error(representative)
    summary = {
        "calibrated_error_test_value": 0.03,
        "max_calibrated_minus_ideal_pseudo_temperature": float(
            np.max(np.abs(calibrated_difference))
        ),
        "representative_parameters": {
            "temperature": representative.temperature,
            "memory_angle": representative.memory_angle,
        },
        "representative_critical_ignored_error": representative_critical,
        "representative_critical_ignored_error_percent": 100.0
        * representative_critical,
        "median_critical_error_percent_nonzero_memory": float(
            100.0 * np.median(critical[:, nonzero_memory])
        ),
        "minimum_positive_critical_error_percent": float(
            100.0 * np.min(critical[:, nonzero_memory])
        ),
        "maximum_critical_error_percent": float(
            100.0 * np.max(critical[:, nonzero_memory])
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
