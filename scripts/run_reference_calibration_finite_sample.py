"""Compare explicit reference calibration with internal self-calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.assignment_fisher import (
    assignment_calibration_fisher,
    assignment_fisher_matrix,
    effective_target_information,
)
from memory_thermometry.classical_full_swap import classical_record_distribution
from memory_thermometry.inference import (
    BlockExperiment,
    CalibrationExperiment,
    calibration_probability_jacobian,
    fisher_scoring_mle,
    quantum_probability_jacobian,
)
from memory_thermometry.model import CollisionParameters


def allocate_total_outcomes(
    total_outcomes: int,
    sensing_fraction: float,
    ground_fraction: float,
    block_length: int,
) -> tuple[int, int, int]:
    """Return sensing blocks, ground shots, and excited shots exactly."""

    sensing_blocks = int(round(sensing_fraction * total_outcomes / block_length))
    ground_shots = int(round(ground_fraction * total_outcomes))
    excited_shots = total_outcomes - block_length * sensing_blocks - ground_shots
    if excited_shots < 1:
        raise ValueError("allocation leaves no excited-reference shots")
    return sensing_blocks, ground_shots, excited_shots


def posterior_summary(
    posterior: np.ndarray,
    temperatures: np.ndarray,
    betas: np.ndarray,
) -> dict[str, object]:
    """Moments and equal-tail intervals of a two-dimensional grid posterior."""

    temperature_mesh, beta_mesh = np.meshgrid(
        temperatures, betas, indexing="xy"
    )
    mean_temperature = float(np.sum(posterior * temperature_mesh))
    mean_beta = float(np.sum(posterior * beta_mesh))
    temperature_marginal = posterior.sum(axis=0)
    beta_marginal = posterior.sum(axis=1)

    def interval(grid: np.ndarray, marginal: np.ndarray) -> list[float]:
        cumulative = np.cumsum(marginal)
        return [
            float(grid[np.searchsorted(cumulative, 0.025)]),
            float(grid[np.searchsorted(cumulative, 0.975)]),
        ]

    variance_temperature = float(
        np.sum(posterior * (temperature_mesh - mean_temperature) ** 2)
    )
    variance_beta = float(np.sum(posterior * (beta_mesh - mean_beta) ** 2))
    covariance = float(
        np.sum(
            posterior
            * (temperature_mesh - mean_temperature)
            * (beta_mesh - mean_beta)
        )
    )
    return {
        "mean_temperature": mean_temperature,
        "temperature_sd": float(np.sqrt(variance_temperature)),
        "temperature_95_interval": interval(
            temperatures, temperature_marginal
        ),
        "mean_false_negative": mean_beta,
        "false_negative_sd": float(np.sqrt(variance_beta)),
        "false_negative_95_interval": interval(betas, beta_marginal),
        "posterior_correlation": float(
            covariance / np.sqrt(variance_temperature * variance_beta)
        ),
    }


def main() -> None:
    output = ROOT / "results" / "reference_calibration_finite_sample"
    output.mkdir(parents=True, exist_ok=True)

    parameters = CollisionParameters(
        temperature=0.90,
        memory_angle=0.50,
        system_memory_angle=0.55,
    )
    true_vector = np.asarray([0.90, 0.50, 0.02, 0.04])
    alpha = true_vector[2]
    beta = true_vector[3]
    length = 8
    full_swap = 0.5 * np.pi

    sensing_block_fisher = assignment_fisher_matrix(
        parameters, length, alpha, beta
    ).matrix
    sensing_rate = sensing_block_fisher / length
    ground_fisher = assignment_calibration_fisher(alpha, beta, 0)
    excited_fisher = assignment_calibration_fisher(alpha, beta, 1)
    ground_grid = np.linspace(0.0, 0.35, 141)
    excited_grid = np.linspace(0.0, 0.10, 81)
    allocation_information = np.full(
        (excited_grid.size, ground_grid.size), np.nan
    )
    best = (-np.inf, 0, 0, np.zeros((4, 4)))
    for i, excited_fraction in enumerate(excited_grid):
        for j, ground_fraction in enumerate(ground_grid):
            sensing_fraction = 1.0 - ground_fraction - excited_fraction
            if sensing_fraction <= 0.0:
                continue
            information = (
                sensing_fraction * sensing_rate
                + ground_fraction * ground_fisher
                + excited_fraction * excited_fisher
            )
            effective = effective_target_information(information)
            allocation_information[i, j] = effective
            if effective > best[0]:
                best = (effective, i, j, information)
    effective_information = float(best[0])
    excited_fraction = float(excited_grid[best[1]])
    ground_fraction = float(ground_grid[best[2]])
    sensing_fraction = 1.0 - ground_fraction - excited_fraction

    sensing_probability, sensing_jacobian = quantum_probability_jacobian(
        true_vector, parameters, full_swap, length
    )
    ground_probability, ground_jacobian = calibration_probability_jacobian(
        true_vector, 0
    )
    excited_probability, excited_jacobian = calibration_probability_jacobian(
        true_vector, 1
    )
    sensing_fisher_check = (
        sensing_jacobian / sensing_probability
    ) @ sensing_jacobian.T
    ground_fisher_check = (
        ground_jacobian / ground_probability
    ) @ ground_jacobian.T
    excited_fisher_check = (
        excited_jacobian / excited_probability
    ) @ excited_jacobian.T

    one_step_total_outcomes = np.asarray(
        [100_000, 300_000, 1_000_000, 3_000_000, 8_000_000, 30_000_000, 80_000_000]
    )
    one_step_replicates = 4000
    one_step_rows: list[dict[str, object]] = []
    one_step_temperatures: dict[str, np.ndarray] = {}
    for total_outcomes in one_step_total_outcomes:
        sensing_blocks, ground_shots, excited_shots = allocate_total_outcomes(
            int(total_outcomes), sensing_fraction, ground_fraction, length
        )
        total_fisher = (
            sensing_blocks * sensing_fisher_check
            + ground_shots * ground_fisher_check
            + excited_shots * excited_fisher_check
        )
        covariance = np.linalg.inv(total_fisher)
        rng = np.random.default_rng(120000 + int(np.log10(total_outcomes)))
        sensing_counts = rng.multinomial(
            sensing_blocks,
            sensing_probability,
            size=one_step_replicates,
        )
        ground_counts = rng.multinomial(
            ground_shots,
            ground_probability,
            size=one_step_replicates,
        )
        excited_counts = rng.multinomial(
            excited_shots,
            excited_probability,
            size=one_step_replicates,
        )
        score = (
            (sensing_counts - sensing_blocks * sensing_probability)
            @ (sensing_jacobian / sensing_probability).T
            + (ground_counts - ground_shots * ground_probability)
            @ (ground_jacobian / ground_probability).T
            + (excited_counts - excited_shots * excited_probability)
            @ (excited_jacobian / excited_probability).T
        )
        estimates = true_vector + score @ covariance.T
        temperature_estimates = estimates[:, 0]
        predicted_sd = float(np.sqrt(covariance[0, 0]))
        empirical_sd = float(np.std(temperature_estimates, ddof=1))
        one_step_temperatures[f"N{int(total_outcomes)}"] = temperature_estimates
        one_step_rows.append(
            {
                "total_binary_readouts": int(total_outcomes),
                "sensing_blocks": sensing_blocks,
                "ground_reference_shots": ground_shots,
                "excited_reference_shots": excited_shots,
                "predicted_temperature_sd": predicted_sd,
                "empirical_temperature_sd": empirical_sd,
                "empirical_to_predicted_sd_ratio": empirical_sd / predicted_sd,
                "temperature_bias": float(
                    np.mean(temperature_estimates - true_vector[0])
                ),
                "wald_95_coverage": float(
                    np.mean(
                        np.abs(temperature_estimates - true_vector[0])
                        <= 1.96 * predicted_sd
                    )
                ),
                "negative_beta_fraction": float(np.mean(estimates[:, 3] < 0.0)),
            }
        )

    mle_total_outcomes = np.asarray([8_000_000, 80_000_000])
    mle_replicates = 32
    bounds = np.asarray(
        [[0.30, 2.00], [0.02, 1.35], [0.001, 0.15], [0.001, 0.35]]
    )
    mle_rows: list[dict[str, object]] = []
    mle_estimates: dict[str, np.ndarray] = {}
    for total_outcomes in mle_total_outcomes:
        sensing_blocks, ground_shots, excited_shots = allocate_total_outcomes(
            int(total_outcomes), sensing_fraction, ground_fraction, length
        )
        total_fisher = (
            sensing_blocks * sensing_fisher_check
            + ground_shots * ground_fisher_check
            + excited_shots * excited_fisher_check
        )
        covariance_prediction = np.linalg.inv(total_fisher)
        predicted_sd = float(np.sqrt(covariance_prediction[0, 0]))
        estimates = np.zeros((mle_replicates, 4))
        standard_errors = np.zeros(mle_replicates)
        converged = np.zeros(mle_replicates, dtype=bool)
        cover = np.zeros(mle_replicates, dtype=bool)
        iterations = np.zeros(mle_replicates, dtype=int)
        for replicate in range(mle_replicates):
            rng = np.random.default_rng(
                140000 + int(np.log10(total_outcomes)) * 100 + replicate
            )
            sensing_counts = rng.multinomial(
                sensing_blocks, sensing_probability
            )
            ground_counts = rng.multinomial(ground_shots, ground_probability)
            excited_counts = rng.multinomial(
                excited_shots, excited_probability
            )
            experiments = [
                BlockExperiment(full_swap, sensing_counts),
                CalibrationExperiment(0, ground_counts),
                CalibrationExperiment(1, excited_counts),
            ]
            score = (
                sensing_jacobian
                @ ((sensing_counts - sensing_blocks * sensing_probability) / sensing_probability)
                + ground_jacobian
                @ ((ground_counts - ground_shots * ground_probability) / ground_probability)
                + excited_jacobian
                @ ((excited_counts - excited_shots * excited_probability) / excited_probability)
            )
            initial = np.clip(
                true_vector + covariance_prediction @ score,
                bounds[:, 0],
                bounds[:, 1],
            )
            fit = fisher_scoring_mle(
                experiments,
                parameters,
                length,
                initial,
                bounds=bounds,
                max_iterations=15,
            )
            estimates[replicate] = fit.estimate
            standard_errors[replicate] = np.sqrt(fit.covariance[0, 0])
            converged[replicate] = fit.converged
            cover[replicate] = (
                abs(fit.estimate[0] - true_vector[0])
                <= 1.96 * standard_errors[replicate]
            )
            iterations[replicate] = fit.iterations
        mle_estimates[f"N{int(total_outcomes)}"] = estimates
        errors = estimates[:, 0] - true_vector[0]
        mle_rows.append(
            {
                "total_binary_readouts": int(total_outcomes),
                "replicates": mle_replicates,
                "predicted_temperature_sd": predicted_sd,
                "empirical_temperature_sd": float(
                    np.std(estimates[:, 0], ddof=1)
                ),
                "temperature_bias": float(np.mean(errors)),
                "temperature_rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_reported_temperature_se": float(
                    np.mean(standard_errors)
                ),
                "wald_95_coverage": float(np.mean(cover)),
                "converged_fraction": float(np.mean(converged)),
                "mean_iterations": float(np.mean(iterations)),
            }
        )
        print(
            f"external calibration, N={int(total_outcomes)}: "
            f"{int(converged.sum())}/{mle_replicates} converged"
        )

    posterior_total_outcomes = 200_000
    sensing_blocks, ground_shots, excited_shots = allocate_total_outcomes(
        posterior_total_outcomes, sensing_fraction, ground_fraction, length
    )
    rng = np.random.default_rng(20260723)
    posterior_sensing_counts = rng.multinomial(
        sensing_blocks, sensing_probability
    )
    posterior_excited_counts = rng.multinomial(
        excited_shots, excited_probability
    )
    temperatures = np.linspace(0.70, 1.35, 131)
    betas = np.linspace(0.0, 0.20, 121)
    sensing_log_likelihood = np.empty((betas.size, temperatures.size))
    for i, candidate_beta in enumerate(betas):
        for j, candidate_temperature in enumerate(temperatures):
            candidate_parameters = CollisionParameters(
                temperature=float(candidate_temperature),
                memory_angle=parameters.memory_angle,
                system_memory_angle=parameters.system_memory_angle,
            )
            probability = classical_record_distribution(
                candidate_parameters,
                length,
                false_positive=alpha,
                false_negative=float(candidate_beta),
            )
            sensing_log_likelihood[i, j] = float(
                np.dot(
                    posterior_sensing_counts,
                    np.log(np.maximum(probability, 1e-300)),
                )
            )
    calibration_log_likelihood = (
        posterior_excited_counts[0] * np.log(np.maximum(betas, 1e-300))
        + posterior_excited_counts[1]
        * np.log(np.maximum(1.0 - betas, 1e-300))
    )
    internal_shifted = sensing_log_likelihood - np.max(sensing_log_likelihood)
    internal_posterior = np.exp(internal_shifted)
    internal_posterior /= internal_posterior.sum()
    external_log_likelihood = (
        sensing_log_likelihood + calibration_log_likelihood[:, None]
    )
    external_shifted = external_log_likelihood - np.max(external_log_likelihood)
    external_posterior = np.exp(external_shifted)
    external_posterior /= external_posterior.sum()
    internal_summary = posterior_summary(internal_posterior, temperatures, betas)
    external_summary = posterior_summary(external_posterior, temperatures, betas)

    phase7_path = ROOT / "results" / "finite_sample_validation" / "summary.json"
    phase7 = json.loads(phase7_path.read_text(encoding="utf-8"))
    internal_comparison = {
        int(row["total_blocks"]) * length: row
        for row in phase7["mle_results"]
        if row["protocol"] == "block_mixture"
    }

    np.savez(
        output / "reference_calibration_data.npz",
        ground_grid=ground_grid,
        excited_grid=excited_grid,
        allocation_information=allocation_information,
        temperatures=temperatures,
        betas=betas,
        internal_posterior=internal_posterior,
        external_posterior=external_posterior,
        posterior_sensing_counts=posterior_sensing_counts,
        posterior_excited_counts=posterior_excited_counts,
        **{
            f"one_step_temperature_{key}": value
            for key, value in one_step_temperatures.items()
        },
        **{
            f"mle_estimates_{key}": value
            for key, value in mle_estimates.items()
        },
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ratio = allocation_information / effective_information
    image = ax.imshow(
        ratio,
        origin="lower",
        aspect="auto",
        extent=[
            100.0 * ground_grid.min(),
            100.0 * ground_grid.max(),
            100.0 * excited_grid.min(),
            100.0 * excited_grid.max(),
        ],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.plot(
        100.0 * ground_fraction,
        100.0 * excited_fraction,
        marker="*",
        color="red",
        markersize=13,
        label="optimal allocation",
    )
    ax.set_xlabel("prepared-ground share (%)")
    ax.set_ylabel("prepared-excited share (%)")
    ax.set_title("N=8 reference-calibration budget design")
    ax.legend()
    fig.colorbar(image, ax=ax, label="fraction of optimum information")
    fig.savefig(output / "reference_allocation_N8.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
    for ax, posterior, title in (
        (axes[0], internal_posterior, "sensing only"),
        (axes[1], external_posterior, "with excited-reference counts"),
    ):
        relative_log10 = np.log10(
            np.maximum(posterior / posterior.max(), 1e-10)
        )
        image = ax.imshow(
            relative_log10,
            origin="lower",
            aspect="auto",
            extent=[
                temperatures.min(),
                temperatures.max(),
                betas.min(),
                betas.max(),
            ],
            vmin=-8.0,
            vmax=0.0,
            cmap="magma",
        )
        ax.plot(true_vector[0], true_vector[3], "co", markersize=5)
        ax.set_xlabel(r"temperature $T/\omega$")
        ax.set_ylabel(r"false-negative rate $\beta$")
        ax.set_title(title)
    fig.colorbar(image, ax=axes, label="log10 posterior / maximum")
    fig.savefig(output / "bayesian_ridge_collapse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.7), constrained_layout=True)
    outcomes = np.asarray([row["total_binary_readouts"] for row in one_step_rows])
    predicted = np.asarray([row["predicted_temperature_sd"] for row in one_step_rows])
    empirical = np.asarray([row["empirical_temperature_sd"] for row in one_step_rows])
    ax.plot(outcomes, predicted, linewidth=2, label="external calibration Fisher")
    ax.scatter(outcomes, empirical, s=32, label="external Monte Carlo")
    internal_outcomes = np.asarray([8_000_000, 80_000_000])
    internal_predicted = np.asarray(
        [
            internal_comparison[int(value)]["predicted_temperature_sd"]
            for value in internal_outcomes
        ]
    )
    ax.scatter(
        internal_outcomes,
        internal_predicted,
        marker="s",
        s=55,
        label="internal block mixture Fisher",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total binary readouts")
    ax.set_ylabel("temperature standard deviation")
    ax.set_title("Reference calibration changes the practical sample scale")
    ax.legend()
    fig.savefig(output / "external_vs_internal_precision.png", dpi=180)
    plt.close(fig)

    comparison_rows = []
    for external_row in mle_rows:
        total = int(external_row["total_binary_readouts"])
        internal_row = internal_comparison[total]
        comparison_rows.append(
            {
                "total_binary_readouts": total,
                "external_temperature_rmse": external_row["temperature_rmse"],
                "internal_temperature_rmse": internal_row["temperature_rmse"],
                "internal_to_external_rmse_ratio": (
                    internal_row["temperature_rmse"]
                    / external_row["temperature_rmse"]
                ),
                "external_converged_fraction": external_row["converged_fraction"],
                "internal_converged_fraction": internal_row["converged_fraction"],
            }
        )
    summary = {
        "parameters": {
            "temperature": parameters.temperature,
            "memory_angle": parameters.memory_angle,
            "false_positive": alpha,
            "false_negative": beta,
            "record_length": length,
        },
        "optimal_readout_allocation": {
            "sensing_fraction": sensing_fraction,
            "ground_reference_fraction": ground_fraction,
            "excited_reference_fraction": excited_fraction,
            "effective_temperature_information_per_readout": effective_information,
        },
        "one_step_replicates": one_step_replicates,
        "one_step_results": one_step_rows,
        "mle_replicates_per_case": mle_replicates,
        "external_calibration_mle_results": mle_rows,
        "external_vs_internal_mle": comparison_rows,
        "conditional_bayesian_posterior": {
            "total_binary_readouts": posterior_total_outcomes,
            "sensing_only": internal_summary,
            "with_reference_calibration": external_summary,
            "conditioning": "mu and alpha fixed at their true values; uniform grid prior",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
