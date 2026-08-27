"""Validate four-parameter Fisher predictions with finite block-count data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.inference import (
    BlockExperiment,
    fisher_scoring_mle,
    quantum_probability_jacobian,
)
from memory_thermometry.model import CollisionParameters


def allocate_blocks(total: int, weights: list[float]) -> list[int]:
    """Allocate an exact integer block budget across protocol settings."""

    allocation = [int(round(total * weight)) for weight in weights]
    allocation[-1] += total - sum(allocation)
    return allocation


def matrix_effective_temperature_information(matrix: np.ndarray) -> float:
    """Efficient information from the inverse covariance identity."""

    inverse = np.linalg.pinv(matrix, rcond=1e-12)
    return float(1.0 / inverse[0, 0])


def main() -> None:
    output = ROOT / "results" / "finite_sample_validation"
    output.mkdir(parents=True, exist_ok=True)

    template = CollisionParameters(
        temperature=0.90,
        memory_angle=0.50,
        system_memory_angle=0.55,
    )
    true_vector = np.asarray([0.90, 0.50, 0.02, 0.04])
    length = 8
    protocols: dict[str, list[tuple[float | tuple[float, ...], float]]] = {
        "best_fixed": [(0.85, 1.0)],
        "alternating": [((0.70, 1.00), 1.0)],
        "block_mixture": [(0.70, 0.66), (0.5 * np.pi, 0.34)],
    }
    design: dict[str, list[dict[str, object]]] = {}
    per_block_information: dict[str, np.ndarray] = {}
    for name, settings in protocols.items():
        components: list[dict[str, object]] = []
        average_information = np.zeros((4, 4))
        for schedule, weight in settings:
            probability, jacobian = quantum_probability_jacobian(
                true_vector,
                template,
                schedule,
                length,
            )
            fisher = (jacobian / probability) @ jacobian.T
            components.append(
                {
                    "schedule": schedule,
                    "weight": weight,
                    "probability": probability,
                    "jacobian": jacobian,
                    "fisher": fisher,
                }
            )
            average_information += weight * fisher
        design[name] = components
        per_block_information[name] = average_information

    one_step_block_counts = np.asarray(
        [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
    )
    one_step_replicates = 4000
    one_step_rows: list[dict[str, object]] = []
    saved_one_step_temperatures: dict[str, np.ndarray] = {}
    for protocol_index, (name, components) in enumerate(design.items()):
        weights = [float(component["weight"]) for component in components]
        for total_blocks in one_step_block_counts:
            allocation = allocate_blocks(int(total_blocks), weights)
            total_information = np.zeros((4, 4))
            scores = np.zeros((one_step_replicates, 4))
            rng = np.random.default_rng(
                71000 + 1000 * protocol_index + int(np.log10(total_blocks))
            )
            for component, block_count in zip(components, allocation):
                probability = np.asarray(component["probability"])
                jacobian = np.asarray(component["jacobian"])
                fisher = np.asarray(component["fisher"])
                counts = rng.multinomial(
                    block_count,
                    probability,
                    size=one_step_replicates,
                )
                centered = counts - block_count * probability
                scores += centered @ (jacobian / probability).T
                total_information += block_count * fisher
            covariance = np.linalg.pinv(total_information, rcond=1e-12)
            estimates = true_vector + scores @ covariance.T
            temperature_estimates = estimates[:, 0]
            predicted_sd = float(np.sqrt(covariance[0, 0]))
            empirical_sd = float(np.std(temperature_estimates, ddof=1))
            valid = (
                (estimates[:, 0] > 0.0)
                & (estimates[:, 1] >= 0.0)
                & (estimates[:, 1] <= 0.5 * np.pi)
                & (estimates[:, 2] >= 0.0)
                & (estimates[:, 3] >= 0.0)
                & (estimates[:, 2] + estimates[:, 3] < 1.0)
            )
            key = f"{name}_B{int(total_blocks)}"
            saved_one_step_temperatures[key] = temperature_estimates
            one_step_rows.append(
                {
                    "protocol": name,
                    "total_blocks": int(total_blocks),
                    "replicates": one_step_replicates,
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
                    "physical_parameter_fraction": float(np.mean(valid)),
                    "negative_beta_fraction": float(np.mean(estimates[:, 3] < 0.0)),
                }
            )

    mle_block_counts = np.asarray([1_000_000, 10_000_000])
    mle_replicates = 16
    bounds = np.asarray(
        [[0.30, 2.00], [0.02, 1.35], [0.001, 0.15], [0.001, 0.35]]
    )
    mle_rows: list[dict[str, object]] = []
    saved_mle_estimates: dict[str, np.ndarray] = {}
    for protocol_index, (name, components) in enumerate(design.items()):
        weights = [float(component["weight"]) for component in components]
        for total_blocks in mle_block_counts:
            allocation = allocate_blocks(int(total_blocks), weights)
            total_information = sum(
                block_count * np.asarray(component["fisher"])
                for component, block_count in zip(components, allocation)
            )
            covariance_prediction = np.linalg.pinv(
                total_information, rcond=1e-12
            )
            predicted_sd = float(np.sqrt(covariance_prediction[0, 0]))
            estimates = np.zeros((mle_replicates, 4))
            standard_errors = np.zeros(mle_replicates)
            converged = np.zeros(mle_replicates, dtype=bool)
            beta_boundary = np.zeros(mle_replicates, dtype=bool)
            cover = np.zeros(mle_replicates, dtype=bool)
            iterations = np.zeros(mle_replicates, dtype=int)
            for replicate in range(mle_replicates):
                rng = np.random.default_rng(
                    93000
                    + 10000 * protocol_index
                    + int(np.log10(total_blocks)) * 100
                    + replicate
                )
                experiments = []
                score = np.zeros(4)
                for component, block_count in zip(components, allocation):
                    probability = np.asarray(component["probability"])
                    jacobian = np.asarray(component["jacobian"])
                    counts = rng.multinomial(block_count, probability)
                    experiments.append(
                        BlockExperiment(component["schedule"], counts)
                    )
                    score += jacobian @ (
                        (counts - block_count * probability) / probability
                    )
                one_step_initial = true_vector + covariance_prediction @ score
                one_step_initial = np.clip(
                    one_step_initial, bounds[:, 0], bounds[:, 1]
                )
                fit = fisher_scoring_mle(
                    experiments,
                    template,
                    length,
                    initial=one_step_initial,
                    bounds=bounds,
                    max_iterations=15,
                )
                estimates[replicate] = fit.estimate
                standard_errors[replicate] = np.sqrt(fit.covariance[0, 0])
                converged[replicate] = fit.converged
                beta_boundary[replicate] = fit.estimate[3] <= bounds[3, 0] + 2e-4
                cover[replicate] = (
                    abs(fit.estimate[0] - true_vector[0])
                    <= 1.96 * standard_errors[replicate]
                )
                iterations[replicate] = fit.iterations
            key = f"{name}_B{int(total_blocks)}"
            saved_mle_estimates[key] = estimates
            temperature_errors = estimates[:, 0] - true_vector[0]
            empirical_sd = float(np.std(estimates[:, 0], ddof=1))
            mle_rows.append(
                {
                    "protocol": name,
                    "total_blocks": int(total_blocks),
                    "replicates": mle_replicates,
                    "predicted_temperature_sd": predicted_sd,
                    "empirical_temperature_sd": empirical_sd,
                    "empirical_to_predicted_sd_ratio": empirical_sd / predicted_sd,
                    "temperature_bias": float(np.mean(temperature_errors)),
                    "temperature_rmse": float(
                        np.sqrt(np.mean(temperature_errors**2))
                    ),
                    "mean_reported_temperature_se": float(
                        np.mean(standard_errors)
                    ),
                    "wald_95_coverage": float(np.mean(cover)),
                    "converged_fraction": float(np.mean(converged)),
                    "beta_lower_boundary_fraction": float(
                        np.mean(beta_boundary)
                    ),
                    "mean_iterations": float(np.mean(iterations)),
                }
            )
            print(
                f"{name}, B={int(total_blocks)}: "
                f"{int(converged.sum())}/{mle_replicates} converged"
            )

    np.savez(
        output / "finite_sample_validation_data.npz",
        true_vector=true_vector,
        one_step_block_counts=one_step_block_counts,
        mle_block_counts=mle_block_counts,
        **{
            f"one_step_temperature_{key}": value
            for key, value in saved_one_step_temperatures.items()
        },
        **{
            f"mle_estimates_{key}": value
            for key, value in saved_mle_estimates.items()
        },
    )

    colors = {
        "best_fixed": "#3A78B4",
        "alternating": "#D17A22",
        "block_mixture": "#3E9B68",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.7), constrained_layout=True)
    for name in protocols:
        selected = [row for row in one_step_rows if row["protocol"] == name]
        blocks = np.asarray([row["total_blocks"] for row in selected])
        predicted = np.asarray(
            [row["predicted_temperature_sd"] for row in selected]
        )
        empirical = np.asarray(
            [row["empirical_temperature_sd"] for row in selected]
        )
        ax.plot(
            blocks,
            predicted,
            color=colors[name],
            linewidth=2,
            label=f"{name.replace('_', ' ')} Fisher",
        )
        ax.scatter(blocks, empirical, color=colors[name], marker="o", s=35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("independent length-8 blocks B")
    ax.set_ylabel("temperature standard deviation")
    ax.set_title("One-step Monte Carlo follows the Fisher prediction")
    ax.legend(fontsize=8)
    fig.savefig(output / "one_step_fisher_validation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    box_data = []
    box_labels = []
    box_colors = []
    for total_blocks in mle_block_counts:
        for name in protocols:
            values = saved_mle_estimates[f"{name}_B{int(total_blocks)}"][:, 0]
            box_data.append(values)
            box_labels.append(
                f"{name.replace('_', ' ')}\n{int(total_blocks / 1e6)}M"
            )
            box_colors.append(colors[name])
    boxplot = ax.boxplot(
        box_data, tick_labels=box_labels, patch_artist=True
    )
    for patch, color in zip(boxplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(true_vector[0], color="black", linestyle="--", label="true T")
    ax.set_ylabel(r"local MLE temperature $\hat T$")
    ax.set_title("Exact-likelihood finite-sample estimates")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend()
    fig.savefig(output / "mle_temperature_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    x = np.arange(len(protocols))
    width = 0.34
    for offset, total_blocks in enumerate(mle_block_counts):
        selected = [
            next(
                row
                for row in mle_rows
                if row["protocol"] == name
                and row["total_blocks"] == int(total_blocks)
            )
            for name in protocols
        ]
        axes[0].bar(
            x + (offset - 0.5) * width,
            [row["empirical_to_predicted_sd_ratio"] for row in selected],
            width,
            label=f"B={int(total_blocks / 1e6)}M",
        )
        axes[1].bar(
            x + (offset - 0.5) * width,
            [row["wald_95_coverage"] for row in selected],
            width,
            label=f"B={int(total_blocks / 1e6)}M",
        )
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[0].set_ylabel("empirical SD / Fisher SD")
    axes[0].set_title("MLE variance convergence")
    axes[1].axhline(0.95, color="black", linestyle="--")
    axes[1].set_ylabel("nominal 95% interval coverage")
    axes[1].set_title("Local Laplace/Wald coverage")
    for ax in axes:
        ax.set_xticks(x, [name.replace("_", "\n") for name in protocols])
        ax.legend(fontsize=8)
    fig.savefig(output / "mle_fisher_diagnostics.png", dpi=180)
    plt.close(fig)

    per_protocol_summary = {
        name: {
            "effective_information_per_block": matrix_effective_temperature_information(
                matrix
            ),
            "predicted_temperature_sd_B1M": float(
                np.sqrt(np.linalg.pinv(1_000_000 * matrix)[0, 0])
            ),
            "predicted_temperature_sd_B10M": float(
                np.sqrt(np.linalg.pinv(10_000_000 * matrix)[0, 0])
            ),
        }
        for name, matrix in per_block_information.items()
    }
    summary = {
        "true_parameters": {
            "temperature": true_vector[0],
            "memory_angle": true_vector[1],
            "false_positive": true_vector[2],
            "false_negative": true_vector[3],
            "system_memory_angle": template.system_memory_angle,
        },
        "record_length": length,
        "one_step_replicates": one_step_replicates,
        "mle_replicates_per_case": mle_replicates,
        "protocols": per_protocol_summary,
        "one_step_results": one_step_rows,
        "mle_results": mle_rows,
        "method_note": (
            "The exact multinomial block likelihood is optimized by bounded "
            "Fisher scoring. Each local MLE is initialized with the one-step "
            "estimate to test local asymptotic Fisher predictions rather than "
            "global-search performance."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
