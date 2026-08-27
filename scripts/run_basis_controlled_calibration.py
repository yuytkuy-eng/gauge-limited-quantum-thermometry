"""Test whether known measurement-basis controls break the detector gauge."""

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
from memory_thermometry.model import CollisionParameters


OUTPUT = ROOT / "results" / "basis_controlled_calibration"
LENGTH = 8
FALSE_POSITIVE = 0.02
FALSE_NEGATIVE = 0.04
PARAMETERS = CollisionParameters(
    temperature=0.9,
    memory_angle=0.5,
    system_memory_angle=0.55,
)


def fisher_for_basis_schedule(schedule: tuple[float, ...]) -> np.ndarray:
    """Return the four-parameter block Fisher matrix for one basis schedule."""

    return quantum_assignment_fisher_matrix(
        PARAMETERS,
        probe_angles=0.5 * np.pi,
        measurement_polar_angles=schedule,
        length=LENGTH,
        false_positive=FALSE_POSITIVE,
        false_negative=FALSE_NEGATIVE,
    ).matrix


def effective_information(matrix: np.ndarray) -> float:
    return effective_target_information(matrix, relative_tolerance=1e-10)


def schedule_label(bits: tuple[int, ...]) -> str:
    return "".join("X" if bit else "Z" for bit in bits)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    z_bits = (0,) * LENGTH
    z_schedule = (0.0,) * LENGTH
    z_fisher = fisher_for_basis_schedule(z_schedule)
    weights = np.linspace(0.0, 1.0, 201)

    rows: list[dict[str, float | int | str]] = []
    best: dict[str, object] | None = None
    for index in range(1, 2**LENGTH):
        bits = tuple(int(bit) for bit in f"{index:0{LENGTH}b}")
        schedule = tuple(0.5 * np.pi if bit else 0.0 for bit in bits)
        candidate_fisher = fisher_for_basis_schedule(schedule)
        pure_information = effective_information(candidate_fisher)

        mixed_information = np.asarray(
            [
                effective_information(
                    weight * candidate_fisher + (1.0 - weight) * z_fisher
                )
                for weight in weights
            ]
        )
        optimum_index = int(np.argmax(mixed_information))
        optimum_weight = float(weights[optimum_index])
        optimum_information = float(mixed_information[optimum_index])
        mixed_fisher = (
            optimum_weight * candidate_fisher
            + (1.0 - optimum_weight) * z_fisher
        )
        row = {
            "schedule": schedule_label(bits),
            "x_shots": int(sum(bits)),
            "pure_effective_information_per_block": pure_information,
            "optimal_schedule_block_fraction": optimum_weight,
            "mixed_effective_information_per_block": optimum_information,
            "mixed_minimum_eigenvalue": float(np.linalg.eigvalsh(mixed_fisher)[0]),
        }
        rows.append(row)
        if best is None or optimum_information > float(best["information"]):
            best = {
                "bits": bits,
                "schedule": schedule_label(bits),
                "weight": optimum_weight,
                "information": optimum_information,
                "pure_information": pure_information,
                "fisher": mixed_fisher,
            }

    assert best is not None
    best_fisher = np.asarray(best["fisher"])
    basis_information_per_readout = float(best["information"]) / LENGTH
    known_detector_information = effective_information(best_fisher[:2, :2])

    internal_summary = json.loads(
        (ROOT / "results" / "probe_gauge_breaking" / "summary.json").read_text()
    )
    internal_information_per_readout = (
        float(
            internal_summary["best_independent_block_mixture"][
                "effective_temperature_information"
            ]
        )
        / LENGTH
    )
    external_summary = json.loads(
        (
            ROOT
            / "results"
            / "reference_calibration_finite_sample"
            / "summary.json"
        ).read_text()
    )
    external_information_per_readout = float(
        external_summary["optimal_readout_allocation"][
            "effective_temperature_information_per_readout"
        ]
    )

    eigenvalues = np.linalg.eigvalsh(best_fisher)
    summary = {
        "parameters": {
            "temperature": PARAMETERS.temperature,
            "memory_angle": PARAMETERS.memory_angle,
            "system_memory_angle": PARAMETERS.system_memory_angle,
            "false_positive": FALSE_POSITIVE,
            "false_negative": FALSE_NEGATIVE,
            "block_length": LENGTH,
        },
        "scan": {
            "basis_alphabet": ["Z", "X"],
            "schedules_tested": 2**LENGTH,
            "mixture_weight_step": float(weights[1] - weights[0]),
            "baseline_schedule": schedule_label(z_bits),
        },
        "optimal_basis_control": {
            "complementary_schedule": best["schedule"],
            "complementary_block_fraction": best["weight"],
            "all_z_block_fraction": 1.0 - float(best["weight"]),
            "pure_complementary_information_per_block": best[
                "pure_information"
            ],
            "effective_information_per_block": best["information"],
            "effective_information_per_readout": basis_information_per_readout,
            "eigenvalues": eigenvalues.tolist(),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "known_detector_information_per_block_same_design": (
                known_detector_information
            ),
            "fraction_of_known_detector_information": (
                float(best["information"]) / known_detector_information
            ),
        },
        "fair_readout_comparison": {
            "previous_internal_non_full_swap_information_per_readout": (
                internal_information_per_readout
            ),
            "basis_control_gain_over_previous_internal": (
                basis_information_per_readout / internal_information_per_readout
            ),
            "external_reference_information_per_readout": (
                external_information_per_readout
            ),
            "external_to_basis_control_information_ratio": (
                external_information_per_readout / basis_information_per_readout
            ),
            "external_to_basis_control_sd_ratio": float(
                np.sqrt(
                    external_information_per_readout
                    / basis_information_per_readout
                )
            ),
        },
        "interpretation": (
            "Known basis rotations break the full-swap detector-temperature "
            "gauge without sacrificing the reset architecture. Z and X block "
            "Fisher matrices are strongly complementary, but explicit state "
            "references remain substantially more efficient at this operating "
            "point."
        ),
    }

    with (OUTPUT / "schedule_scan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    labels = ["non-full-swap\ninternal", "basis-controlled\ninternal", "external\nreferences"]
    values = [
        internal_information_per_readout,
        basis_information_per_readout,
        external_information_per_readout,
    ]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    bars = axis.bar(labels, values, color=["#8c96c6", "#2b8cbe", "#e34a33"])
    axis.set_yscale("log")
    axis.set_ylabel("effective temperature information per readout")
    axis.set_title("Self-calibration improves with basis control")
    axis.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + 0.5 * bar.get_width(),
            value * 1.18,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT / "information_comparison.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
