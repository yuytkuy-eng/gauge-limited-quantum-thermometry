"""Measure the gauge-breaking exponent of small readout-basis rotations."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.assignment_fisher import (
    effective_target_information,
    quantum_assignment_fisher_matrix,
)
from memory_thermometry.identifiability import fit_power_law
from memory_thermometry.model import CollisionParameters, record_distribution


OUTPUT = ROOT / "results" / "basis_symmetry_scaling"
PARAMETERS = CollisionParameters(
    temperature=0.9,
    memory_angle=0.5,
    system_memory_angle=0.55,
)
FALSE_POSITIVE = 0.02
FALSE_NEGATIVE = 0.04
LENGTH = 8
PHIS = np.geomspace(0.008, 0.35, 18)
FIT_POINTS = 8


def fisher(polar_angles: float | tuple[float, ...]) -> np.ndarray:
    return quantum_assignment_fisher_matrix(
        PARAMETERS,
        probe_angles=0.5 * np.pi,
        measurement_polar_angles=polar_angles,
        length=LENGTH,
        false_positive=FALSE_POSITIVE,
        false_negative=FALSE_NEGATIVE,
    ).matrix


def evenness_diagnostic(
    polar_plus: float | tuple[float, ...],
    azimuth_plus: float | tuple[float, ...],
    polar_minus: float | tuple[float, ...],
    azimuth_minus: float | tuple[float, ...],
    step: float,
) -> dict[str, float]:
    plus = record_distribution(
        PARAMETERS,
        0.5 * np.pi,
        LENGTH,
        false_positive=FALSE_POSITIVE,
        false_negative=FALSE_NEGATIVE,
        measurement_polar_angles=polar_plus,
        measurement_azimuths=azimuth_plus,
    )
    minus = record_distribution(
        PARAMETERS,
        0.5 * np.pi,
        LENGTH,
        false_positive=FALSE_POSITIVE,
        false_negative=FALSE_NEGATIVE,
        measurement_polar_angles=polar_minus,
        measurement_azimuths=azimuth_minus,
    )
    derivative = (plus - minus) / (2.0 * step)
    return {
        "l2_norm": float(np.linalg.norm(derivative)),
        "maximum_absolute_component": float(np.max(np.abs(derivative))),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    z_fisher = fisher(0.0)
    rows: list[dict[str, float]] = []

    for phi in PHIS:
        fixed_fisher = fisher(float(phi))
        alternating_fisher = fisher((0.0, float(phi)))
        mixed_fisher = 0.5 * z_fisher + 0.5 * fixed_fisher
        rows.append(
            {
                "basis_angle": float(phi),
                "fixed_effective_information": effective_target_information(
                    fixed_fisher
                ),
                "alternating_effective_information": (
                    effective_target_information(alternating_fisher)
                ),
                "block_mixture_effective_information": (
                    effective_target_information(mixed_fisher)
                ),
                "fixed_minimum_eigenvalue": float(
                    np.linalg.eigvalsh(fixed_fisher)[0]
                ),
                "alternating_minimum_eigenvalue": float(
                    np.linalg.eigvalsh(alternating_fisher)[0]
                ),
                "block_mixture_minimum_eigenvalue": float(
                    np.linalg.eigvalsh(mixed_fisher)[0]
                ),
            }
        )

    control = np.asarray([row["basis_angle"] for row in rows[:FIT_POINTS]])
    fit_fields = {
        "fixed_effective_information": "fixed",
        "alternating_effective_information": "alternating",
        "block_mixture_effective_information": "block_mixture",
        "fixed_minimum_eigenvalue": "fixed_minimum_eigenvalue",
        "alternating_minimum_eigenvalue": "alternating_minimum_eigenvalue",
        "block_mixture_minimum_eigenvalue": "block_mixture_minimum_eigenvalue",
    }
    fits: dict[str, dict[str, float]] = {}
    for field, label in fit_fields.items():
        fit = fit_power_law(
            control,
            [row[field] for row in rows[:FIT_POINTS]],
        )
        fits[label] = {
            "exponent": fit.exponent,
            "coefficient": fit.coefficient,
            "r_squared": fit.r_squared,
        }

    test_phi = 0.01
    tilted_fisher = fisher(test_phi)
    weights = np.linspace(0.0, 1.0, 201)
    weight_information = np.asarray(
        [
            effective_target_information(
                (1.0 - weight) * z_fisher + weight * tilted_fisher
            )
            for weight in weights
        ]
    )
    best_weight_index = int(np.argmax(weight_information))

    symmetry_step = 1e-3
    symmetry = {
        "fixed": evenness_diagnostic(
            symmetry_step,
            0.0,
            symmetry_step,
            np.pi,
            symmetry_step,
        ),
        "alternating": evenness_diagnostic(
            (0.0, symmetry_step),
            (0.0, 0.0),
            (0.0, symmetry_step),
            (0.0, np.pi),
            symmetry_step,
        ),
    }

    old_summary = json.loads(
        (ROOT / "results" / "probe_gauge_breaking" / "summary.json").read_text()
    )
    summary = {
        "parameters": {
            "temperature": PARAMETERS.temperature,
            "memory_angle": PARAMETERS.memory_angle,
            "system_memory_angle": PARAMETERS.system_memory_angle,
            "false_positive": FALSE_POSITIVE,
            "false_negative": FALSE_NEGATIVE,
            "block_length": LENGTH,
        },
        "small_angle_fit": {
            "points": FIT_POINTS,
            "angle_interval": [float(control[0]), float(control[-1])],
            "fits": fits,
        },
        "sign_reversal_symmetry": {
            "finite_difference_step": symmetry_step,
            **symmetry,
        },
        "asymptotic_block_design": {
            "test_angle": test_phi,
            "optimal_tilted_block_fraction": float(weights[best_weight_index]),
            "effective_information": float(
                weight_information[best_weight_index]
            ),
            "quartic_coefficient": float(
                weight_information[best_weight_index] / test_phi**4
            ),
        },
        "comparison": {
            "previous_incomplete_reset_information_exponent": old_summary[
                "full_swap_boundary_scaling"
            ]["effective_information_exponent"],
            "basis_block_mixture_information_exponent": fits["block_mixture"][
                "exponent"
            ],
        },
        "interpretation": (
            "Excitation-number symmetry makes every record probability even "
            "under a common sign reversal of the real basis rotation. The "
            "observable model therefore leaves the detector-temperature "
            "gauge only at second order in the angle, and efficient Fisher "
            "information appears at fourth order."
        ),
    }

    with (OUTPUT / "basis_scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    colors = {
        "fixed": "#756bb1",
        "alternating": "#2b8cbe",
        "block_mixture": "#e34a33",
    }
    labels = {
        "fixed": "fixed tilted basis",
        "alternating": "Z/tilted within block",
        "block_mixture": "50% Z + 50% tilted blocks",
    }
    for key in ("fixed", "alternating", "block_mixture"):
        axes[0].loglog(
            PHIS,
            [row[f"{key}_effective_information"] for row in rows],
            "o-",
            ms=3.5,
            color=colors[key],
            label=labels[key],
        )
        axes[1].loglog(
            PHIS,
            [row[f"{key}_minimum_eigenvalue"] for row in rows],
            "o-",
            ms=3.5,
            color=colors[key],
            label=labels[key],
        )
    guide = rows[0]["block_mixture_effective_information"] * (
        PHIS / PHIS[0]
    ) ** 4
    axes[0].loglog(PHIS, guide, "k--", lw=1.2, label=r"$\phi^4$")
    eigen_guide = rows[0]["block_mixture_minimum_eigenvalue"] * (
        PHIS / PHIS[0]
    ) ** 4
    axes[1].loglog(PHIS, eigen_guide, "k--", lw=1.2, label=r"$\phi^4$")
    axes[0].set_ylabel("effective temperature information")
    axes[1].set_ylabel("minimum Fisher eigenvalue")
    for axis in axes:
        axis.set_xlabel(r"basis rotation angle $\phi$")
        axis.grid(which="both", alpha=0.22)
        axis.legend(fontsize=8)
    axes[0].set_title("Quotient score leaves the gauge quadratically")
    axes[1].set_title("The singular Fisher direction lifts quartically")
    figure.tight_layout()
    figure.savefig(OUTPUT / "basis_quartic_scaling.png", dpi=220)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
