"""Test whether non-full-swap probe readout breaks the detector gauge."""

from __future__ import annotations

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


def known_detector_information(matrix: np.ndarray) -> float:
    """Temperature information after eliminating memory but not detector data."""

    physical = matrix[:2, :2]
    return float(
        physical[0, 0] - physical[0, 1] ** 2 / physical[1, 1]
    )


def main() -> None:
    output = ROOT / "results" / "probe_gauge_breaking"
    output.mkdir(parents=True, exist_ok=True)

    parameters = CollisionParameters(
        temperature=0.90,
        memory_angle=0.50,
        system_memory_angle=0.55,
    )
    alpha = 0.02
    beta = 0.04
    length = 8
    rank_tolerance = 1e-12

    angles = np.concatenate(
        [np.linspace(0.15, 1.50, 28), np.asarray([0.5 * np.pi])]
    )
    fixed_matrices = []
    fixed_effective = []
    fixed_known = []
    fixed_eigenvalues = []
    fixed_ranks = []
    for angle in angles:
        result = quantum_assignment_fisher_matrix(
            parameters,
            float(angle),
            length,
            alpha,
            beta,
            rank_tolerance=rank_tolerance,
        )
        fixed_matrices.append(result.matrix)
        fixed_effective.append(result.effective_temperature_information)
        fixed_known.append(known_detector_information(result.matrix))
        fixed_eigenvalues.append(result.eigenvalues)
        fixed_ranks.append(result.rank)
    fixed_effective = np.asarray(fixed_effective)
    fixed_known = np.asarray(fixed_known)
    fixed_eigenvalues = np.asarray(fixed_eigenvalues)
    fixed_ranks = np.asarray(fixed_ranks)
    best_fixed_index = int(np.argmax(fixed_effective))

    mixture_weights = np.linspace(0.02, 0.98, 49)
    pair_effective = np.zeros((angles.size, angles.size))
    pair_weight_first = np.full_like(pair_effective, 0.5)
    best_mixture = (-np.inf, 0, 0, 0.5, np.zeros((4, 4)))
    for i in range(angles.size):
        pair_effective[i, i] = fixed_effective[i]
        for j in range(i + 1, angles.size):
            local_best = -np.inf
            local_weight = 0.5
            local_matrix = np.zeros((4, 4))
            for weight in mixture_weights:
                combined = (
                    weight * fixed_matrices[i]
                    + (1.0 - weight) * fixed_matrices[j]
                )
                effective = effective_target_information(
                    combined, relative_tolerance=rank_tolerance
                )
                if effective > local_best:
                    local_best = effective
                    local_weight = float(weight)
                    local_matrix = combined
            pair_effective[i, j] = local_best
            pair_effective[j, i] = local_best
            pair_weight_first[i, j] = local_weight
            pair_weight_first[j, i] = 1.0 - local_weight
            if local_best > best_mixture[0]:
                best_mixture = (
                    local_best,
                    i,
                    j,
                    local_weight,
                    local_matrix,
                )

    alternating_first = np.linspace(0.50, 0.80, 7)
    alternating_second = np.linspace(0.90, 1.20, 7)
    alternating_effective = np.zeros(
        (alternating_first.size, alternating_second.size)
    )
    best_alternating = (-np.inf, 0, 0, np.zeros((4, 4)))
    for i, first in enumerate(alternating_first):
        for j, second in enumerate(alternating_second):
            result = quantum_assignment_fisher_matrix(
                parameters,
                (float(first), float(second)),
                length,
                alpha,
                beta,
                rank_tolerance=rank_tolerance,
            )
            alternating_effective[i, j] = result.effective_temperature_information
            if result.effective_temperature_information > best_alternating[0]:
                best_alternating = (
                    result.effective_temperature_information,
                    i,
                    j,
                    result.matrix,
                )

    endpoint_offsets = np.asarray([0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.28])
    endpoint_effective = []
    endpoint_minimum_eigenvalue = []
    for offset in endpoint_offsets:
        result = quantum_assignment_fisher_matrix(
            parameters,
            0.5 * np.pi - float(offset),
            length,
            alpha,
            beta,
            rank_tolerance=1e-14,
        )
        endpoint_effective.append(result.effective_temperature_information)
        endpoint_minimum_eigenvalue.append(result.eigenvalues[0])
    endpoint_effective = np.asarray(endpoint_effective)
    endpoint_minimum_eigenvalue = np.asarray(endpoint_minimum_eigenvalue)
    effective_exponent = float(
        np.polyfit(
            np.log(endpoint_offsets[:5]),
            np.log(endpoint_effective[:5]),
            1,
        )[0]
    )
    eigenvalue_exponent = float(
        np.polyfit(
            np.log(endpoint_offsets[:5]),
            np.log(endpoint_minimum_eigenvalue[:5]),
            1,
        )[0]
    )

    derivative_scales = np.asarray([0.5, 1.0, 2.0, 4.0])
    derivative_stability = []
    best_fixed_angle = float(angles[best_fixed_index])
    for scale in derivative_scales:
        result = quantum_assignment_fisher_matrix(
            parameters,
            best_fixed_angle,
            length,
            alpha,
            beta,
            temperature_step=1e-4 * float(scale),
            memory_step=1e-4 * float(scale),
            error_step=1e-5 * float(scale),
            rank_tolerance=rank_tolerance,
        )
        derivative_stability.append(result.effective_temperature_information)
    derivative_stability = np.asarray(derivative_stability)

    length_values = np.asarray([4, 6, 8, 10])
    fixed_by_length = []
    mixture_by_length = []
    mixture_first_angle = float(angles[best_mixture[1]])
    mixture_second_angle = float(angles[best_mixture[2]])
    mixture_first_weight = float(best_mixture[3])
    for candidate_length in length_values:
        fixed_result = quantum_assignment_fisher_matrix(
            parameters,
            best_fixed_angle,
            int(candidate_length),
            alpha,
            beta,
            rank_tolerance=rank_tolerance,
        )
        first_result = quantum_assignment_fisher_matrix(
            parameters,
            mixture_first_angle,
            int(candidate_length),
            alpha,
            beta,
            rank_tolerance=rank_tolerance,
        )
        second_result = quantum_assignment_fisher_matrix(
            parameters,
            mixture_second_angle,
            int(candidate_length),
            alpha,
            beta,
            rank_tolerance=rank_tolerance,
        )
        mixed_matrix = (
            mixture_first_weight * first_result.matrix
            + (1.0 - mixture_first_weight) * second_result.matrix
        )
        fixed_by_length.append(fixed_result.effective_temperature_information)
        mixture_by_length.append(
            effective_target_information(
                mixed_matrix, relative_tolerance=rank_tolerance
            )
        )
    fixed_by_length = np.asarray(fixed_by_length)
    mixture_by_length = np.asarray(mixture_by_length)

    np.savez(
        output / "probe_gauge_breaking_data.npz",
        angles=angles,
        fixed_effective=fixed_effective,
        fixed_known=fixed_known,
        fixed_eigenvalues=fixed_eigenvalues,
        fixed_ranks=fixed_ranks,
        pair_effective=pair_effective,
        pair_weight_first=pair_weight_first,
        alternating_first=alternating_first,
        alternating_second=alternating_second,
        alternating_effective=alternating_effective,
        endpoint_offsets=endpoint_offsets,
        endpoint_effective=endpoint_effective,
        endpoint_minimum_eigenvalue=endpoint_minimum_eigenvalue,
        derivative_scales=derivative_scales,
        derivative_stability=derivative_stability,
        length_values=length_values,
        fixed_by_length=fixed_by_length,
        mixture_by_length=mixture_by_length,
    )

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 7.2), constrained_layout=True)
    axes[0].plot(angles, fixed_known, label="known detector", linewidth=2)
    axes[0].plot(
        angles,
        fixed_effective,
        label=r"unknown $\mu,\alpha,\beta$",
        linewidth=2,
    )
    axes[0].set_yscale("log")
    axes[0].axvline(0.5 * np.pi, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(best_fixed_angle, color="tab:red", linestyle=":", linewidth=1.5)
    axes[0].set_ylabel("effective temperature information")
    axes[0].set_title("Non-full-swap readout breaks the exact gauge weakly")
    axes[0].legend()
    eigenvalue_ratio = np.maximum(
        np.abs(fixed_eigenvalues[:, 0] / fixed_eigenvalues[:, -1]), 1e-20
    )
    axes[1].plot(angles, eigenvalue_ratio, "o-", markersize=3)
    axes[1].axvline(0.5 * np.pi, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"probe readout angle $\theta$")
    axes[1].set_ylabel(r"$|\lambda_{\min}|/\lambda_{\max}$")
    fig.savefig(output / "fixed_angle_gauge_breaking.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1, 4.8), constrained_layout=True)
    ax.loglog(
        endpoint_offsets,
        endpoint_effective,
        "o-",
        label=fr"effective information, slope {effective_exponent:.2f}",
    )
    ax.loglog(
        endpoint_offsets,
        endpoint_minimum_eigenvalue,
        "s-",
        label=fr"minimum eigenvalue, slope {eigenvalue_exponent:.2f}",
    )
    reference = endpoint_effective[0] * (endpoint_offsets / endpoint_offsets[0]) ** 4
    ax.loglog(endpoint_offsets, reference, "k--", label=r"quartic $\epsilon^4$")
    ax.set_xlabel(r"distance from full swap $\epsilon=\pi/2-\theta$")
    ax.set_ylabel("gauge-breaking information")
    ax.set_title("Gauge breaking vanishes quartically near full swap")
    ax.legend()
    fig.savefig(output / "quartic_gauge_breaking.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    pair_gain = pair_effective / fixed_effective[best_fixed_index]
    image = ax.imshow(
        pair_gain,
        origin="lower",
        aspect="auto",
        extent=[angles.min(), angles.max(), angles.min(), angles.max()],
        cmap="viridis",
    )
    ax.plot(
        mixture_second_angle,
        mixture_first_angle,
        marker="*",
        markersize=13,
        color="red",
        label="best block mixture",
    )
    ax.set_xlabel(r"second fixed angle $\theta_2$")
    ax.set_ylabel(r"first fixed angle $\theta_1$")
    ax.set_title("Independent fixed-angle blocks are complementary")
    ax.legend(loc="upper left")
    fig.colorbar(image, ax=ax, label="gain over best fixed angle")
    fig.savefig(output / "mixed_block_gain.png", dpi=180)
    plt.close(fig)

    best_alternating_matrix = best_alternating[3]
    protocol_effective = np.asarray(
        [
            fixed_effective[best_fixed_index],
            best_alternating[0],
            best_mixture[0],
        ]
    )
    protocol_known = np.asarray(
        [
            fixed_known[best_fixed_index],
            known_detector_information(best_alternating_matrix),
            known_detector_information(best_mixture[4]),
        ]
    )
    labels = ["best fixed", "alternating", "block mixture"]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2), constrained_layout=True)
    gain = protocol_effective / protocol_effective[0]
    bars = axes[0].bar(labels, gain, color="#3A78B4")
    axes[0].set_ylabel("gain over best fixed")
    axes[0].set_title("Relative self-calibration gain")
    for bar, value in zip(bars, gain):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}x",
            ha="center",
        )
    retained = 100.0 * protocol_effective / protocol_known
    bars = axes[1].bar(labels, retained, color="#D17A22")
    axes[1].set_ylabel("known-detector information retained (%)")
    axes[1].set_title("Absolute identifiability remains weak")
    for bar, value in zip(bars, retained):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}%",
            ha="center",
        )
    fig.savefig(output / "protocol_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    ax.plot(
        length_values,
        fixed_by_length / length_values,
        "o-",
        label="best fixed",
    )
    ax.plot(
        length_values,
        mixture_by_length / length_values,
        "o-",
        label="optimized block mixture",
    )
    ax.set_xlabel("record length N")
    ax.set_ylabel("effective information per outcome")
    ax.set_title("Complementary block advantage persists with record length")
    ax.legend()
    fig.savefig(output / "length_scaling.png", dpi=180)
    plt.close(fig)

    alternating_angles = [
        float(alternating_first[best_alternating[1]]),
        float(alternating_second[best_alternating[2]]),
    ]
    summary = {
        "parameters": {
            "temperature": parameters.temperature,
            "memory_angle": parameters.memory_angle,
            "system_memory_angle": parameters.system_memory_angle,
            "false_positive": alpha,
            "false_negative": beta,
            "record_length": length,
        },
        "rank_result": {
            "all_non_full_swap_angles_full_rank": bool(
                np.all(fixed_ranks[:-1] == 4)
            ),
            "full_swap_rank": int(fixed_ranks[-1]),
        },
        "best_fixed": {
            "angle": best_fixed_angle,
            "effective_temperature_information": float(
                fixed_effective[best_fixed_index]
            ),
            "known_detector_information": float(fixed_known[best_fixed_index]),
            "fraction_of_known_detector_information": float(
                fixed_effective[best_fixed_index] / fixed_known[best_fixed_index]
            ),
            "condition_number": float(
                fixed_eigenvalues[best_fixed_index, -1]
                / fixed_eigenvalues[best_fixed_index, 0]
            ),
        },
        "best_alternating": {
            "angles": alternating_angles,
            "effective_temperature_information": float(best_alternating[0]),
            "gain_over_best_fixed": float(
                best_alternating[0] / fixed_effective[best_fixed_index]
            ),
            "fraction_of_known_detector_information": float(
                best_alternating[0]
                / known_detector_information(best_alternating_matrix)
            ),
        },
        "best_independent_block_mixture": {
            "angles": [mixture_first_angle, mixture_second_angle],
            "first_angle_weight": mixture_first_weight,
            "second_angle_weight": 1.0 - mixture_first_weight,
            "effective_temperature_information": float(best_mixture[0]),
            "gain_over_best_fixed": float(
                best_mixture[0] / fixed_effective[best_fixed_index]
            ),
            "fraction_of_known_detector_information": float(
                best_mixture[0] / known_detector_information(best_mixture[4])
            ),
        },
        "full_swap_boundary_scaling": {
            "effective_information_exponent": effective_exponent,
            "minimum_eigenvalue_exponent": eigenvalue_exponent,
        },
        "finite_difference_stability": {
            "step_scales": derivative_scales.tolist(),
            "effective_information": derivative_stability.tolist(),
            "relative_range": float(
                (derivative_stability.max() - derivative_stability.min())
                / derivative_stability[1]
            ),
        },
        "interpretation": (
            "Non-full-swap readout breaks the exact gauge, but internal "
            "self-calibration retains below one percent of the corresponding "
            "known-detector temperature information in this scan."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
