"""Compare the joint-output QFI with practical thermometric readouts."""

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
    assignment_fisher_matrix,
    effective_target_information,
)
from memory_thermometry.fisher import classical_fisher_matrix
from memory_thermometry.model import CollisionParameters
from memory_thermometry.quantum_output import full_swap_probe_quantum_fisher


OUTPUT = ROOT / "results" / "quantum_output_benchmark"
PARAMETERS = CollisionParameters(
    temperature=0.9,
    memory_angle=0.5,
    system_memory_angle=0.55,
)
FALSE_POSITIVE = 0.02
FALSE_NEGATIVE = 0.04
MAX_LENGTH = 8


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for length in range(1, MAX_LENGTH + 1):
        quantum = full_swap_probe_quantum_fisher(PARAMETERS, length)
        ideal_energy = classical_fisher_matrix(
            PARAMETERS,
            probe_angles=0.5 * np.pi,
            length=length,
        )
        noisy = assignment_fisher_matrix(
            PARAMETERS,
            length=length,
            false_positive=FALSE_POSITIVE,
            false_negative=FALSE_NEGATIVE,
        )
        known_noisy_information = effective_target_information(
            noisy.matrix[:2, :2]
        )
        qfi = quantum.effective_temperature_information
        rows.append(
            {
                "length": length,
                "state_dimension": quantum.state_dimension,
                "quantum_effective_information": qfi,
                "ideal_energy_effective_information": (
                    ideal_energy.effective_temperature_information
                ),
                "known_noisy_energy_effective_information": (
                    known_noisy_information
                ),
                "unknown_detector_energy_effective_information": (
                    noisy.effective_temperature_information
                ),
                "ideal_energy_fraction_of_qfi": (
                    ideal_energy.effective_temperature_information / qfi
                ),
                "known_noisy_fraction_of_qfi": known_noisy_information / qfi,
                "maximum_mean_sld_commutator": float(
                    np.max(np.abs(quantum.mean_sld_commutator))
                ),
                "minimum_output_state_eigenvalue": (
                    quantum.minimum_state_eigenvalue
                ),
            }
        )

    basis_summary = json.loads(
        (
            ROOT
            / "results"
            / "basis_controlled_calibration"
            / "summary.json"
        ).read_text()
    )
    reference_summary = json.loads(
        (
            ROOT
            / "results"
            / "reference_calibration_finite_sample"
            / "summary.json"
        ).read_text()
    )
    reset_summary = json.loads(
        (ROOT / "results" / "probe_gauge_breaking" / "summary.json").read_text()
    )

    final = rows[-1]
    qfi_per_readout = float(final["quantum_effective_information"]) / MAX_LENGTH
    ideal_per_readout = (
        float(final["ideal_energy_effective_information"]) / MAX_LENGTH
    )
    known_noisy_per_readout = (
        float(final["known_noisy_energy_effective_information"]) / MAX_LENGTH
    )
    external_per_readout = float(
        reference_summary["optimal_readout_allocation"][
            "effective_temperature_information_per_readout"
        ]
    )
    basis_per_readout = float(
        basis_summary["optimal_basis_control"][
            "effective_information_per_readout"
        ]
    )
    reset_per_readout = (
        float(
            reset_summary["best_independent_block_mixture"][
                "effective_temperature_information"
            ]
        )
        / MAX_LENGTH
    )

    ladder = {
        "joint_output_qfi": qfi_per_readout,
        "ideal_energy_measurement": ideal_per_readout,
        "known_noisy_energy_measurement": known_noisy_per_readout,
        "external_reference_calibration": external_per_readout,
        "basis_controlled_internal_calibration": basis_per_readout,
        "incomplete_reset_internal_calibration": reset_per_readout,
        "unknown_detector_all_z": float(
            final["unknown_detector_energy_effective_information"]
        )
        / MAX_LENGTH,
    }
    fractions = {
        name: value / qfi_per_readout for name, value in ladder.items()
    }
    summary = {
        "parameters": {
            "temperature": PARAMETERS.temperature,
            "memory_angle": PARAMETERS.memory_angle,
            "system_memory_angle": PARAMETERS.system_memory_angle,
            "false_positive": FALSE_POSITIVE,
            "false_negative": FALSE_NEGATIVE,
            "maximum_block_length": MAX_LENGTH,
        },
        "length_8_quantum_fisher": {
            "effective_information_per_block": final[
                "quantum_effective_information"
            ],
            "effective_information_per_readout": qfi_per_readout,
            "maximum_mean_sld_commutator": final[
                "maximum_mean_sld_commutator"
            ],
            "weak_compatibility_at_numerical_precision": bool(
                float(final["maximum_mean_sld_commutator"]) < 1e-10
            ),
        },
        "information_ladder_per_readout": ladder,
        "fraction_of_joint_output_qfi": fractions,
        "interpretation": (
            "At N=8, ideal local energy measurement extracts 56.2% of the "
            "effective joint-output QFI and known assignment errors reduce "
            "this to 34.3%. Unknown detector parameters create a qualitatively "
            "larger loss: the all-Z protocol becomes exactly nonidentifiable. "
            "External references recover 21.7% of the QFI rate, whereas the "
            "best tested internal basis control recovers about 0.19%."
        ),
    }

    with (OUTPUT / "length_scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.35))
    lengths = np.arange(1, MAX_LENGTH + 1)
    axes[0].plot(
        lengths,
        [float(row["quantum_effective_information"]) / int(row["length"]) for row in rows],
        "o-",
        label="joint-output QFI",
        color="#252525",
    )
    axes[0].plot(
        lengths,
        [float(row["ideal_energy_effective_information"]) / int(row["length"]) for row in rows],
        "o-",
        label="ideal energy readout",
        color="#2b8cbe",
    )
    axes[0].plot(
        lengths,
        [float(row["known_noisy_energy_effective_information"]) / int(row["length"]) for row in rows],
        "o-",
        label="known noisy readout",
        color="#756bb1",
    )
    axes[0].set_xlabel("record length")
    axes[0].set_ylabel("effective information per probe")
    axes[0].set_title("Measurement restriction creates a growing gap")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    bar_names = [
        "QFI",
        "ideal\nenergy",
        "known\nnoisy",
        "external\nrefs",
        "basis\ncontrol",
        "incomplete\nreset",
    ]
    bar_values = [
        qfi_per_readout,
        ideal_per_readout,
        known_noisy_per_readout,
        external_per_readout,
        basis_per_readout,
        reset_per_readout,
    ]
    bars = axes[1].bar(
        bar_names,
        bar_values,
        color=["#252525", "#2b8cbe", "#756bb1", "#e34a33", "#31a354", "#9e9ac8"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("effective information per binary readout")
    axes[1].set_title("Process, measurement, and calibration losses")
    axes[1].grid(axis="y", which="both", alpha=0.25)
    axes[1].set_ylim(min(bar_values) * 0.65, max(bar_values) * 1.55)
    for bar, value in zip(bars, bar_values, strict=True):
        axes[1].text(
            bar.get_x() + 0.5 * bar.get_width(),
            value * 1.15,
            f"{100.0 * value / qfi_per_readout:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    figure.tight_layout()
    figure.savefig(OUTPUT / "quantum_information_ladder.png", dpi=220)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
