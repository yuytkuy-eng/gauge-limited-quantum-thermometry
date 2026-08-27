"""Map Markov-temperature bias under asymmetric assignment errors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.classical_full_swap import (
    critical_ignored_false_positive_rate,
    critical_ignored_readout_error,
    pseudo_true_markov_temperature,
    pseudo_true_temperature_with_assignment_error,
)
from memory_thermometry.model import CollisionParameters


def main() -> None:
    output = ROOT / "results" / "asymmetric_readout_error"
    output.mkdir(parents=True, exist_ok=True)

    representative = CollisionParameters(temperature=0.90, memory_angle=0.50)
    false_positives = np.linspace(0.0, 0.05, 151)
    false_negatives = np.linspace(0.0, 0.05, 151)
    bias = np.empty((false_negatives.size, false_positives.size))

    for i, beta in enumerate(false_negatives):
        for j, alpha in enumerate(false_positives):
            fitted = pseudo_true_temperature_with_assignment_error(
                representative,
                true_false_positive=float(alpha),
                true_false_negative=float(beta),
                assumed_false_positive=0.0,
                assumed_false_negative=0.0,
            )
            bias[i, j] = fitted - representative.temperature

    critical_curve = np.asarray(
        [
            critical_ignored_false_positive_rate(representative, float(beta))
            for beta in false_negatives
        ]
    )

    temperatures = np.linspace(0.40, 2.40, 41)
    memories = np.linspace(0.0, 1.30, 53)
    fixed_false_negative = 0.01
    critical_surface = np.empty((temperatures.size, memories.size))
    calibrated_difference = np.empty_like(critical_surface)
    calibration_alpha = 0.02
    calibration_beta = 0.04
    for i, temperature in enumerate(temperatures):
        for j, memory in enumerate(memories):
            parameters = CollisionParameters(float(temperature), float(memory))
            critical_surface[i, j] = critical_ignored_false_positive_rate(
                parameters, fixed_false_negative
            )
            ideal = pseudo_true_markov_temperature(parameters)
            calibrated = pseudo_true_temperature_with_assignment_error(
                parameters,
                true_false_positive=calibration_alpha,
                true_false_negative=calibration_beta,
                assumed_false_positive=calibration_alpha,
                assumed_false_negative=calibration_beta,
            )
            calibrated_difference[i, j] = calibrated - ideal

    np.savez(
        output / "asymmetric_readout_error_data.npz",
        representative_temperature=representative.temperature,
        representative_memory_angle=representative.memory_angle,
        false_positives=false_positives,
        false_negatives=false_negatives,
        bias=bias,
        critical_curve=critical_curve,
        temperatures=temperatures,
        memories=memories,
        fixed_false_negative=fixed_false_negative,
        critical_surface=critical_surface,
        calibrated_difference=calibrated_difference,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    image = ax.imshow(
        bias,
        origin="lower",
        aspect="auto",
        extent=[
            100.0 * false_positives.min(),
            100.0 * false_positives.max(),
            100.0 * false_negatives.min(),
            100.0 * false_negatives.max(),
        ],
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=float(np.min(bias)),
            vcenter=0.0,
            vmax=float(np.max(bias)),
        ),
    )
    ax.contour(
        100.0 * false_positives,
        100.0 * false_negatives,
        bias,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.plot(
        100.0 * critical_curve,
        100.0 * false_negatives,
        color="white",
        linestyle="--",
        linewidth=1.2,
        label=r"analytic $\alpha_c(\beta)$",
    )
    ax.set_xlabel(r"false positive $\alpha$ (%)")
    ax.set_ylabel(r"false negative $\beta$ (%)")
    ax.set_title(r"Ignored asymmetric errors at $T/\omega=0.9,\ \mu=0.5$")
    ax.legend(loc="upper left", framealpha=0.9)
    colorbar = fig.colorbar(
        image, ax=ax, label=r"pseudo-temperature bias $T_*-T$"
    )
    colorbar.set_ticks([-0.1, 0.0, 0.4, 0.8, 1.2, 1.5])
    fig.savefig(output / "bias_plane.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.plot(
        100.0 * false_negatives,
        100.0 * critical_curve,
        linewidth=2.0,
    )
    ax.fill_between(
        100.0 * false_negatives,
        0.0,
        100.0 * critical_curve,
        alpha=0.18,
        label="memory/false-negative cooling dominates",
    )
    ax.set_xlabel(r"false negative $\beta$ (%)")
    ax.set_ylabel(r"critical false positive $\alpha_c$ (%)")
    ax.set_title("False positives required to cancel the downward bias")
    ax.legend(frameon=False)
    fig.savefig(output / "critical_false_positive_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    image = ax.imshow(
        100.0 * critical_surface,
        origin="lower",
        aspect="auto",
        extent=[
            memories.min(),
            memories.max(),
            temperatures.min(),
            temperatures.max(),
        ],
    )
    ax.set_xlabel(r"memory angle $\mu$")
    ax.set_ylabel(r"temperature $T/\omega$")
    ax.set_title(r"Critical $\alpha$ when ignored $\beta=1\%$")
    fig.colorbar(image, ax=ax, label=r"critical false positive $\alpha_c$ (%)")
    fig.savefig(output / "critical_surface_beta_1pct.png", dpi=180)
    plt.close(fig)

    representative_thresholds = {
        f"beta_{100.0 * beta:.0f}_percent": float(
            critical_ignored_false_positive_rate(representative, beta)
        )
        for beta in (0.0, 0.01, 0.03)
    }
    symmetric_threshold = critical_ignored_readout_error(representative)
    summary = {
        "representative_parameters": {
            "temperature": representative.temperature,
            "memory_angle": representative.memory_angle,
            "system_memory_angle": representative.system_memory_angle,
        },
        "critical_false_positive_rates": representative_thresholds,
        "critical_false_positive_percent": {
            key: 100.0 * value for key, value in representative_thresholds.items()
        },
        "symmetric_threshold_consistency": {
            "symmetric_rate": symmetric_threshold,
            "general_boundary_alpha_at_beta_equal_symmetric_rate": (
                critical_ignored_false_positive_rate(
                    representative, symmetric_threshold
                )
            ),
        },
        "calibrated_assignment_test": {
            "false_positive": calibration_alpha,
            "false_negative": calibration_beta,
            "max_absolute_pseudo_temperature_change": float(
                np.max(np.abs(calibrated_difference))
            ),
        },
        "bias_plane_fraction_negative": float(np.mean(bias < 0.0)),
        "bias_plane_fraction_positive": float(np.mean(bias > 0.0)),
        "bias_plane_range": [float(np.min(bias)), float(np.max(bias))],
        "critical_surface_fixed_false_negative": fixed_false_negative,
        "critical_surface_median_percent": float(
            100.0 * np.median(critical_surface)
        ),
        "critical_surface_range_percent": [
            float(100.0 * np.min(critical_surface)),
            float(100.0 * np.max(critical_surface)),
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
