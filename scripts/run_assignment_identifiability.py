"""Diagnose detector-temperature nonidentifiability and calibration recovery."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.assignment_fisher import (
    assignment_calibration_fisher,
    assignment_fisher_matrix,
    effective_target_information,
)
from memory_thermometry.classical_full_swap import (
    classical_record_distribution,
    thermal_excitation_probability,
)
from memory_thermometry.model import CollisionParameters


def known_detector_temperature_information(matrix: np.ndarray) -> float:
    """Temperature information after eliminating only memory strength."""

    physical = matrix[:2, :2]
    return float(
        physical[0, 0] - physical[0, 1] ** 2 / physical[1, 1]
    )


def optimize_calibration_allocation(
    sensing_rate: np.ndarray,
    ground_calibration: np.ndarray,
    excited_calibration: np.ndarray,
    ground_fractions: np.ndarray,
    excited_fractions: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Search reference-state allocations at fixed total shot budget."""

    information = np.full(
        (excited_fractions.size, ground_fractions.size), np.nan
    )
    best_value = -np.inf
    best_index = (0, 0)
    for i, excited_fraction in enumerate(excited_fractions):
        for j, ground_fraction in enumerate(ground_fractions):
            sensing_fraction = 1.0 - ground_fraction - excited_fraction
            if sensing_fraction <= 0.0:
                continue
            combined = (
                sensing_fraction * sensing_rate
                + ground_fraction * ground_calibration
                + excited_fraction * excited_calibration
            )
            value = effective_target_information(combined)
            information[i, j] = value
            if value > best_value:
                best_value = value
                best_index = (i, j)
    return information, best_index


def main() -> None:
    output = ROOT / "results" / "assignment_identifiability"
    output.mkdir(parents=True, exist_ok=True)

    parameters = CollisionParameters(
        temperature=0.90,
        memory_angle=0.50,
        system_memory_angle=0.55,
    )
    alpha = 0.02
    beta = 0.04
    lengths = np.arange(3, 13)
    fisher_matrices = []
    eigenvalues = []
    ranks = []
    effective_unknown = []
    for length in lengths:
        result = assignment_fisher_matrix(
            parameters,
            int(length),
            false_positive=alpha,
            false_negative=beta,
        )
        fisher_matrices.append(result.matrix)
        eigenvalues.append(result.eigenvalues)
        ranks.append(result.rank)
        effective_unknown.append(result.effective_temperature_information)
    eigenvalues_array = np.asarray(eigenvalues)

    final_fisher = fisher_matrices[-1]
    final_eigenvalues, final_eigenvectors = np.linalg.eigh(final_fisher)
    numerical_null = final_eigenvectors[:, 0]
    numerical_null *= np.sign(numerical_null[0])
    numerical_null /= numerical_null[0]
    excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    contrast = 1.0 - alpha - beta
    analytic_slope = (
        contrast
        * parameters.energy_gap
        * (1.0 - excitation)
        / parameters.temperature**2
    )
    analytic_null = np.array([1.0, 0.0, 0.0, analytic_slope])

    alternative_temperature = 1.30
    alternative_excitation = thermal_excitation_probability(
        alternative_temperature, parameters.energy_gap
    )
    alternative_beta = 1.0 - alpha - contrast * excitation / alternative_excitation
    coupling_angles = np.array([0.25, 0.55, 0.95, 1.35])
    gauge_differences = []
    multi_setting_fisher = np.zeros((4, 4))
    for coupling in coupling_angles:
        original_parameters = replace(
            parameters, system_memory_angle=float(coupling)
        )
        alternative_parameters = replace(
            original_parameters, temperature=alternative_temperature
        )
        original_distribution = classical_record_distribution(
            original_parameters,
            length=10,
            false_positive=alpha,
            false_negative=beta,
        )
        alternative_distribution = classical_record_distribution(
            alternative_parameters,
            length=10,
            false_positive=alpha,
            false_negative=alternative_beta,
        )
        gauge_differences.append(
            float(np.max(np.abs(original_distribution - alternative_distribution)))
        )
        multi_setting_fisher += assignment_fisher_matrix(
            original_parameters,
            length=10,
            false_positive=alpha,
            false_negative=beta,
        ).matrix / coupling_angles.size
    multi_setting_eigenvalues = np.linalg.eigvalsh(multi_setting_fisher)
    multi_setting_rank = int(
        np.sum(
            multi_setting_eigenvalues
            > 1e-10 * multi_setting_eigenvalues[-1]
        )
    )

    ridge_temperatures = np.linspace(0.85, 1.80, 96)
    ridge_false_negatives = np.linspace(0.0, 0.45, 91)
    reference_distribution = classical_record_distribution(
        parameters,
        length=8,
        false_positive=alpha,
        false_negative=beta,
    )
    likelihood_divergence = np.empty(
        (ridge_false_negatives.size, ridge_temperatures.size)
    )
    for i, candidate_beta in enumerate(ridge_false_negatives):
        for j, candidate_temperature in enumerate(ridge_temperatures):
            candidate = classical_record_distribution(
                replace(parameters, temperature=float(candidate_temperature)),
                length=8,
                false_positive=alpha,
                false_negative=float(candidate_beta),
            )
            likelihood_divergence[i, j] = float(
                np.sum(
                    reference_distribution
                    * np.log(reference_distribution / candidate)
                )
            )
    analytic_ridge_beta = (
        1.0
        - alpha
        - contrast
        * excitation
        / np.asarray(
            [
                thermal_excitation_probability(
                    float(temperature), parameters.energy_gap
                )
                for temperature in ridge_temperatures
            ]
        )
    )

    block_length = int(lengths[-1])
    sensing_rate = final_fisher / block_length
    ground_calibration = assignment_calibration_fisher(
        alpha, beta, prepared_state=0
    )
    excited_calibration = assignment_calibration_fisher(
        alpha, beta, prepared_state=1
    )
    known_detector_information = known_detector_temperature_information(
        sensing_rate
    )
    ground_fractions = np.linspace(0.0, 0.35, 141)
    excited_fractions = np.linspace(0.0, 0.10, 81)
    allocation_information, best_index = optimize_calibration_allocation(
        sensing_rate,
        ground_calibration,
        excited_calibration,
        ground_fractions,
        excited_fractions,
    )
    best_excited_fraction = float(excited_fractions[best_index[0]])
    best_ground_fraction = float(ground_fractions[best_index[1]])
    best_information = float(allocation_information[best_index])

    one_dimensional_fractions = np.linspace(0.0, 0.60, 601)
    bright_only_information = np.asarray(
        [
            effective_target_information(
                (1.0 - fraction) * sensing_rate
                + fraction * excited_calibration
            )
            for fraction in one_dimensional_fractions
        ]
    )
    balanced_information = np.asarray(
        [
            effective_target_information(
                (1.0 - fraction) * sensing_rate
                + 0.5 * fraction * ground_calibration
                + 0.5 * fraction * excited_calibration
            )
            for fraction in one_dimensional_fractions
        ]
    )
    bright_index = int(np.argmax(bright_only_information))
    balanced_index = int(np.argmax(balanced_information))

    np.savez(
        output / "assignment_identifiability_data.npz",
        lengths=lengths,
        eigenvalues=eigenvalues_array,
        ranks=np.asarray(ranks),
        effective_unknown=np.asarray(effective_unknown),
        analytic_null=analytic_null,
        numerical_null=numerical_null,
        coupling_angles=coupling_angles,
        gauge_differences=np.asarray(gauge_differences),
        multi_setting_fisher=multi_setting_fisher,
        ridge_temperatures=ridge_temperatures,
        ridge_false_negatives=ridge_false_negatives,
        likelihood_divergence=likelihood_divergence,
        analytic_ridge_beta=analytic_ridge_beta,
        ground_fractions=ground_fractions,
        excited_fractions=excited_fractions,
        allocation_information=allocation_information,
        one_dimensional_fractions=one_dimensional_fractions,
        bright_only_information=bright_only_information,
        balanced_information=balanced_information,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    normalized_spectrum = np.maximum(
        np.abs(eigenvalues_array) / eigenvalues_array[:, -1, None], 1e-18
    )
    for index in range(4):
        ax.plot(
            lengths,
            normalized_spectrum[:, index],
            "o-",
            label=fr"$\lambda_{index + 1}$",
        )
    ax.set_yscale("log")
    ax.set_xlabel("record length N")
    ax.set_ylabel("eigenvalue / largest eigenvalue")
    ax.set_title(r"Four-parameter Fisher matrix has rank three")
    ax.legend(ncol=2)
    fig.savefig(output / "fisher_rank_deficiency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.7), constrained_layout=True)
    positive_divergence = likelihood_divergence[likelihood_divergence > 1e-14]
    image = ax.imshow(
        np.maximum(likelihood_divergence, 1e-14),
        origin="lower",
        aspect="auto",
        extent=[
            ridge_temperatures.min(),
            ridge_temperatures.max(),
            ridge_false_negatives.min(),
            ridge_false_negatives.max(),
        ],
        norm=LogNorm(
            vmin=1e-14,
            vmax=float(np.max(positive_divergence)),
        ),
        cmap="magma",
    )
    valid_ridge = (
        (analytic_ridge_beta >= ridge_false_negatives.min())
        & (analytic_ridge_beta <= ridge_false_negatives.max())
    )
    ax.plot(
        ridge_temperatures[valid_ridge],
        analytic_ridge_beta[valid_ridge],
        color="cyan",
        linestyle="--",
        linewidth=1.8,
        label=r"constant $(1-\alpha-\beta)e(T)$",
    )
    ax.plot(
        parameters.temperature,
        beta,
        marker="o",
        color="white",
        markeredgecolor="black",
        label="reference point",
    )
    ax.set_xlabel(r"temperature $T/\omega$")
    ax.set_ylabel(r"false-negative rate $\beta$")
    ax.set_title("Exact likelihood ridge from detector-temperature gauge")
    ax.legend(loc="upper left")
    fig.colorbar(image, ax=ax, label="record-distribution KL divergence")
    fig.savefig(output / "likelihood_gauge_ridge.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ratio = allocation_information / known_detector_information
    positive = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    plot_ratio = np.ma.masked_less_equal(ratio, 1e-8)
    allocation_cmap = plt.get_cmap("viridis").copy()
    allocation_cmap.set_bad("#eeeeee")
    image = ax.imshow(
        plot_ratio,
        origin="lower",
        aspect="auto",
        extent=[
            100.0 * ground_fractions.min(),
            100.0 * ground_fractions.max(),
            100.0 * excited_fractions.min(),
            100.0 * excited_fractions.max(),
        ],
        vmin=0.0,
        vmax=1.02 * best_information / known_detector_information,
        cmap=allocation_cmap,
    )
    ax.plot(
        100.0 * best_ground_fraction,
        100.0 * best_excited_fraction,
        marker="*",
        markersize=13,
        color="red",
        label="optimal allocation",
    )
    ax.set_xlabel(r"prepared-ground calibration share $f_0$ (%)")
    ax.set_ylabel(r"prepared-excited calibration share $f_1$ (%)")
    ax.set_title("Recovered temperature information per total shot")
    ax.legend(loc="upper right")
    fig.colorbar(
        image,
        ax=ax,
        label="fraction of known-detector temperature information",
    )
    fig.savefig(output / "calibration_allocation.png", dpi=180)
    plt.close(fig)

    protocol_names = [
        "no\ncalibration",
        "excited\nonly",
        "balanced\nreferences",
        "optimized\nreferences",
        "known\ndetector",
    ]
    protocol_ratios = [
        0.0,
        float(bright_only_information[bright_index] / known_detector_information),
        float(balanced_information[balanced_index] / known_detector_information),
        best_information / known_detector_information,
        1.0,
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    bars = ax.bar(protocol_names, protocol_ratios, color="#3A78B4")
    bars[-1].set_color("#666666")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("effective temperature information ratio")
    ax.set_title("Reference shots restore identifiability")
    for bar, value in zip(bars, protocol_ratios):
        ax.text(
            bar.get_x() + 0.5 * bar.get_width(),
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.savefig(output / "calibration_protocol_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "parameters": {
            "temperature": parameters.temperature,
            "memory_angle": parameters.memory_angle,
            "system_memory_angle": parameters.system_memory_angle,
            "false_positive": alpha,
            "false_negative": beta,
        },
        "fisher_rank_by_length": {
            str(int(length)): int(rank) for length, rank in zip(lengths, ranks)
        },
        "smallest_to_largest_eigenvalue_at_N12": float(
            abs(final_eigenvalues[0]) / final_eigenvalues[-1]
        ),
        "analytic_null_direction_T_normalized": analytic_null.tolist(),
        "numerical_null_direction_T_normalized": numerical_null.tolist(),
        "null_direction_max_absolute_difference": float(
            np.max(np.abs(analytic_null - numerical_null))
        ),
        "gauge_test": {
            "alternative_temperature": alternative_temperature,
            "alternative_false_negative": alternative_beta,
            "constant_contrast_times_excitation": contrast * excitation,
            "maximum_record_probability_difference": float(
                np.max(gauge_differences)
            ),
        },
        "multi_coupling_test": {
            "coupling_angles": coupling_angles.tolist(),
            "combined_fisher_rank": multi_setting_rank,
            "combined_effective_temperature_information": (
                effective_target_information(multi_setting_fisher)
            ),
        },
        "calibration_design_N12": {
            "known_detector_temperature_information_per_sensing_shot": (
                known_detector_information
            ),
            "optimized_ground_reference_fraction": best_ground_fraction,
            "optimized_excited_reference_fraction": best_excited_fraction,
            "optimized_sensing_fraction": (
                1.0 - best_ground_fraction - best_excited_fraction
            ),
            "optimized_effective_information_per_total_shot": best_information,
            "optimized_fraction_of_known_detector_information": (
                best_information / known_detector_information
            ),
            "excited_only_optimal_fraction": float(
                one_dimensional_fractions[bright_index]
            ),
            "excited_only_fraction_of_known_detector_information": float(
                bright_only_information[bright_index]
                / known_detector_information
            ),
            "balanced_optimal_total_fraction": float(
                one_dimensional_fractions[balanced_index]
            ),
            "balanced_fraction_of_known_detector_information": float(
                balanced_information[balanced_index]
                / known_detector_information
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
