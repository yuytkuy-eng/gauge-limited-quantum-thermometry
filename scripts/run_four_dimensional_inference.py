"""Full bounded four-dimensional MLE and Bayesian finite-sample validation."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.inference import (
    BlockExperiment,
    CalibrationExperiment,
    calibration_probability_jacobian,
    fisher_scoring_mle,
    quantum_probability_jacobian,
)
from memory_thermometry.model import CollisionParameters
from memory_thermometry.posterior import (
    BoundedUniformPrior,
    gauge_log_absolute_jacobian,
    gauge_to_physical_coordinates,
    log_posterior,
    physical_to_gauge_coordinates,
    physical_to_gauge_jacobian,
    posterior_summary,
    run_random_walk_metropolis,
)


OUTPUT = ROOT / "results" / "four_dimensional_inference"
PARAMETERS = CollisionParameters(
    temperature=0.9,
    memory_angle=0.5,
    system_memory_angle=0.55,
)
TRUE_VECTOR = np.asarray([0.9, 0.5, 0.02, 0.04])
LENGTH = 8
BOUNDS = np.asarray(
    [[0.30, 2.00], [0.02, 1.35], [0.001, 0.15], [0.001, 0.35]]
)
PRIOR = BoundedUniformPrior(BOUNDS)
PARAMETER_NAMES = ("temperature", "memory_angle", "false_positive", "false_negative")
GAUGE_NAMES = ("temperature", "memory_angle", "false_positive", "kappa")
INTERNAL_X_FRACTION = 0.325
EXTERNAL_FRACTIONS = (0.795, 0.185, 0.020)
TOTAL_READOUTS = (1_000_000, 8_000_000)
MLE_REPLICATES = 96
POSTERIOR_COVERAGE_REPLICATES = 1000
POSTERIOR_DRAWS = 5000
POSTERIOR_BURN_IN = 2000

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def binomial_wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Wilson 95% interval for an estimated coverage probability."""

    z = 1.959963984540054
    fraction = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (fraction + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def allocate_internal(total_readouts: int) -> tuple[int, int]:
    total_blocks = total_readouts // LENGTH
    x_blocks = int(round(INTERNAL_X_FRACTION * total_blocks))
    return total_blocks - x_blocks, x_blocks


def allocate_external(total_readouts: int) -> tuple[int, int, int]:
    sensing_fraction, ground_fraction, _ = EXTERNAL_FRACTIONS
    sensing_blocks = int(round(total_readouts * sensing_fraction / LENGTH))
    ground_shots = int(round(total_readouts * ground_fraction))
    excited_shots = total_readouts - LENGTH * sensing_blocks - ground_shots
    return sensing_blocks, ground_shots, excited_shots


def information(probability: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    return (jacobian / probability) @ jacobian.T


def build_designs() -> dict[str, dict[str, object]]:
    z_probability, z_jacobian = quantum_probability_jacobian(
        TRUE_VECTOR,
        PARAMETERS,
        0.5 * np.pi,
        LENGTH,
        measurement_polar_angles=0.0,
    )
    x_probability, x_jacobian = quantum_probability_jacobian(
        TRUE_VECTOR,
        PARAMETERS,
        0.5 * np.pi,
        LENGTH,
        measurement_polar_angles=0.5 * np.pi,
    )
    ground_probability, ground_jacobian = calibration_probability_jacobian(
        TRUE_VECTOR, 0
    )
    excited_probability, excited_jacobian = calibration_probability_jacobian(
        TRUE_VECTOR, 1
    )
    return {
        "internal_basis": {
            "components": (
                ("z", z_probability, z_jacobian),
                ("x", x_probability, x_jacobian),
            ),
        },
        "external_references": {
            "components": (
                ("z", z_probability, z_jacobian),
                ("ground", ground_probability, ground_jacobian),
                ("excited", excited_probability, excited_jacobian),
            ),
        },
    }


def allocations(protocol: str, total_readouts: int) -> tuple[int, ...]:
    if protocol == "internal_basis":
        return allocate_internal(total_readouts)
    if protocol == "external_references":
        return allocate_external(total_readouts)
    raise ValueError("unknown protocol")


def experiments_from_counts(
    protocol: str, counts: list[np.ndarray]
) -> list[BlockExperiment | CalibrationExperiment]:
    if protocol == "internal_basis":
        return [
            BlockExperiment(0.5 * np.pi, counts[0], 0.0),
            BlockExperiment(0.5 * np.pi, counts[1], 0.5 * np.pi),
        ]
    return [
        BlockExperiment(0.5 * np.pi, counts[0], 0.0),
        CalibrationExperiment(0, counts[1]),
        CalibrationExperiment(1, counts[2]),
    ]


def simulate_counts(
    design: dict[str, object], allocation: tuple[int, ...], rng: np.random.Generator
) -> list[np.ndarray]:
    components = design["components"]
    return [
        rng.multinomial(count, np.asarray(component[1]))
        for component, count in zip(components, allocation, strict=True)
    ]


def fisher_for_design(
    design: dict[str, object], allocation: tuple[int, ...]
) -> np.ndarray:
    matrix = np.zeros((4, 4))
    for component, count in zip(
        design["components"], allocation, strict=True
    ):
        matrix += count * information(
            np.asarray(component[1]), np.asarray(component[2])
        )
    return matrix


def initial_from_score(
    design: dict[str, object], allocation: tuple[int, ...], counts: list[np.ndarray]
) -> np.ndarray:
    fisher = fisher_for_design(design, allocation)
    score = np.zeros(4)
    for component, count, observed in zip(
        design["components"], allocation, counts, strict=True
    ):
        probability = np.asarray(component[1])
        jacobian = np.asarray(component[2])
        score += jacobian @ ((observed - count * probability) / probability)
    initial = np.clip(
        TRUE_VECTOR + np.linalg.pinv(fisher, rcond=1e-12) @ score,
        BOUNDS[:, 0],
        BOUNDS[:, 1],
    )
    if initial[2] + initial[3] >= PRIOR.maximum_assignment_sum:
        initial = TRUE_VECTOR.copy()
    return initial


def run_mle_ensemble(designs: dict[str, dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {}
    for protocol_index, (protocol, design) in enumerate(designs.items()):
        for total_readouts in TOTAL_READOUTS:
            allocation = allocations(protocol, total_readouts)
            fisher = fisher_for_design(design, allocation)
            predicted_covariance = np.linalg.inv(fisher)
            predicted_sd = np.sqrt(np.diag(predicted_covariance))
            estimates = np.empty((MLE_REPLICATES, 4))
            standard_errors = np.empty((MLE_REPLICATES, 4))
            converged = np.zeros(MLE_REPLICATES, dtype=bool)
            boundary = np.zeros((MLE_REPLICATES, 4), dtype=bool)
            rng = np.random.default_rng(
                131000 + 1000 * protocol_index + int(np.log10(total_readouts))
            )
            for replicate in range(MLE_REPLICATES):
                counts = simulate_counts(design, allocation, rng)
                experiments = experiments_from_counts(protocol, counts)
                fit = fisher_scoring_mle(
                    experiments,
                    PARAMETERS,
                    LENGTH,
                    initial_from_score(design, allocation, counts),
                    bounds=BOUNDS,
                    max_iterations=20,
                )
                estimates[replicate] = fit.estimate
                standard_errors[replicate] = np.sqrt(
                    np.maximum(np.diag(fit.covariance), 0.0)
                )
                converged[replicate] = fit.converged
                boundary[replicate] = (
                    (fit.estimate - BOUNDS[:, 0] < 2e-4)
                    | (BOUNDS[:, 1] - fit.estimate < 2e-4)
                )
            saved[f"{protocol}_N{total_readouts}"] = estimates
            for parameter, name in enumerate(PARAMETER_NAMES):
                errors = estimates[:, parameter] - TRUE_VECTOR[parameter]
                cover = np.abs(errors) <= 1.96 * standard_errors[:, parameter]
                rows.append(
                    {
                        "protocol": protocol,
                        "total_binary_readouts": total_readouts,
                        "independent_units": int(sum(allocation)),
                        "parameter": name,
                        "replicates": MLE_REPLICATES,
                        "predicted_sd": float(predicted_sd[parameter]),
                        "empirical_sd": float(
                            np.std(estimates[:, parameter], ddof=1)
                        ),
                        "bias": float(np.mean(errors)),
                        "rmse": float(np.sqrt(np.mean(errors**2))),
                        "mean_reported_se": float(
                            np.mean(standard_errors[:, parameter])
                        ),
                        "wald_95_coverage": float(np.mean(cover)),
                        "wald_coverage_monte_carlo_95_interval": list(
                            binomial_wilson_interval(
                                int(np.count_nonzero(cover)), MLE_REPLICATES
                            )
                        ),
                        "boundary_fraction": float(
                            np.mean(boundary[:, parameter])
                        ),
                        "converged_fraction": float(np.mean(converged)),
                    }
                )
    return rows, saved


def covariance_coverage(
    design: dict[str, object], allocation: tuple[int, ...], seed: int
) -> dict[str, object]:
    fisher = fisher_for_design(design, allocation)
    covariance = np.linalg.inv(fisher)
    rng = np.random.default_rng(seed)
    score = np.zeros((POSTERIOR_COVERAGE_REPLICATES, 4))
    for component, count in zip(
        design["components"], allocation, strict=True
    ):
        probability = np.asarray(component[1])
        jacobian = np.asarray(component[2])
        counts = rng.multinomial(
            count, probability, size=POSTERIOR_COVERAGE_REPLICATES
        )
        score += (counts - count * probability) @ (
            jacobian / probability
        ).T
    estimates = TRUE_VECTOR + score @ covariance.T
    sd = np.sqrt(np.diag(covariance))
    cover = np.abs(estimates - TRUE_VECTOR) <= 1.96 * sd
    physical = (
        np.all(estimates >= BOUNDS[:, 0], axis=1)
        & np.all(estimates <= BOUNDS[:, 1], axis=1)
        & (estimates[:, 2] + estimates[:, 3] < PRIOR.maximum_assignment_sum)
    )
    return {
        "replicates": POSTERIOR_COVERAGE_REPLICATES,
        "linearized_95_coverage": np.mean(cover, axis=0).tolist(),
        "linearized_bias": np.mean(estimates - TRUE_VECTOR, axis=0).tolist(),
        "linearized_empirical_sd": np.std(estimates, axis=0, ddof=1).tolist(),
        "fisher_predicted_sd": sd.tolist(),
        "physical_parameter_fraction": float(np.mean(physical)),
        "negative_assignment_fraction": float(
            np.mean((estimates[:, 2] < 0.0) | (estimates[:, 3] < 0.0))
        ),
    }


def posterior_initials(
    center: np.ndarray, covariance: np.ndarray, seed: int
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, np.max(eigenvalues) * 1e-8)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(4):
        for _ in range(1000):
            proposal = center + 0.8 * root @ rng.normal(size=4)
            if PRIOR.contains(proposal):
                starts.append(proposal)
                break
        else:
            starts.append(center.copy())
    return np.asarray(starts)


def run_posterior_case(
    protocol: str,
    design: dict[str, object],
    total_readouts: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    allocation = allocations(protocol, total_readouts)
    rng = np.random.default_rng(seed)
    counts = simulate_counts(design, allocation, rng)
    experiments = experiments_from_counts(protocol, counts)
    initial = initial_from_score(design, allocation, counts)
    fit = fisher_scoring_mle(
        experiments,
        PARAMETERS,
        LENGTH,
        initial,
        bounds=BOUNDS,
        max_iterations=24,
    )
    covariance = 0.5 * (fit.covariance + fit.covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = eigenvectors @ np.diag(
        np.maximum(eigenvalues, np.max(eigenvalues) * 1e-8)
    ) @ eigenvectors.T

    if protocol == "external_references":
        initials = posterior_initials(fit.estimate, covariance, seed + 1)
        result = run_random_walk_metropolis(
            lambda vector: log_posterior(
                vector, PARAMETERS, experiments, LENGTH, PRIOR
            ),
            initials,
            1.4 * covariance,
            draws=POSTERIOR_DRAWS,
            burn_in=POSTERIOR_BURN_IN,
            seed=seed + 2,
        )
        samples = result.samples
        coordinate_system = "physical (T,mu,alpha,beta)"
    else:
        center = physical_to_gauge_coordinates(fit.estimate)
        jacobian = physical_to_gauge_jacobian(fit.estimate)
        gauge_covariance = jacobian @ covariance @ jacobian.T
        gauge_initials = np.asarray(
            [physical_to_gauge_coordinates(value) for value in posterior_initials(fit.estimate, covariance, seed + 1)]
        )

        def gauge_density(gauge_vector: np.ndarray) -> float:
            physical = gauge_to_physical_coordinates(gauge_vector)
            return log_posterior(
                physical, PARAMETERS, experiments, LENGTH, PRIOR
            ) + gauge_log_absolute_jacobian(gauge_vector)

        result = run_random_walk_metropolis(
            gauge_density,
            gauge_initials,
            1.4 * gauge_covariance,
            draws=POSTERIOR_DRAWS,
            burn_in=POSTERIOR_BURN_IN,
            seed=seed + 2,
        )
        samples = np.empty_like(result.samples)
        for chain in range(samples.shape[0]):
            for draw in range(samples.shape[1]):
                samples[chain, draw] = gauge_to_physical_coordinates(
                    result.samples[chain, draw]
                )
        coordinate_system = "gauge-adapted (T,mu,alpha,kappa), reported in physical coordinates"

    summary = posterior_summary(samples)
    coverage = (
        (summary.interval_95[:, 0] <= TRUE_VECTOR)
        & (TRUE_VECTOR <= summary.interval_95[:, 1])
    )
    record = {
        "protocol": protocol,
        "total_binary_readouts": total_readouts,
        "allocation": list(allocation),
        "independent_units": int(sum(allocation)),
        "mle": fit.estimate.tolist(),
        "mle_converged": bool(fit.converged),
        "coordinate_system": coordinate_system,
        "posterior_mean": summary.mean.tolist(),
        "posterior_median": summary.median.tolist(),
        "posterior_sd": summary.standard_deviation.tolist(),
        "equal_tailed_95_interval": summary.interval_95.tolist(),
        "true_value_covered": coverage.tolist(),
        "correlation": summary.correlation.tolist(),
        "acceptance_rate": result.acceptance_rate.tolist(),
        "split_rhat_sampling_coordinates": result.split_rhat.tolist(),
        "bulk_ess_sampling_coordinates": result.bulk_effective_sample_size.tolist(),
        "proposal_scales": result.proposal_scales.tolist(),
        "draws_per_chain": POSTERIOR_DRAWS,
        "chains": 4,
        "burn_in_per_chain": POSTERIOR_BURN_IN,
        "prior": {
            "type": "uniform on bounded physical domain",
            "bounds": BOUNDS.tolist(),
            "maximum_alpha_plus_beta": PRIOR.maximum_assignment_sum,
        },
    }
    return record, samples, np.asarray(counts, dtype=object)


def pair_plot(
    samples: dict[str, np.ndarray], posterior_records: list[dict[str, object]]
) -> None:
    labels = [r"$T$", r"$\mu$", r"$\alpha$", r"$\beta$"]
    colors = {"internal_basis": "#2b8cbe", "external_references": "#e34a33"}
    figure, axes = plt.subplots(4, 4, figsize=(7.2, 6.8))
    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            for record in posterior_records:
                key = record["protocol"]
                values = samples[key].reshape(-1, 4)[::4]
                if row == column:
                    axis.hist(
                        values[:, column],
                        bins=45,
                        density=True,
                        histtype="step",
                        linewidth=1.5,
                        color=colors[key],
                        label=key.replace("_", " ").capitalize(),
                    )
                    axis.axvline(TRUE_VECTOR[column], color="black", linestyle="--", linewidth=1)
                else:
                    axis.scatter(
                        values[:, column],
                        values[:, row],
                        s=2,
                        alpha=0.09,
                        color=colors[key],
                        rasterized=True,
                    )
                    axis.plot(TRUE_VECTOR[column], TRUE_VECTOR[row], "k+", ms=7)
            if row == 3:
                axis.set_xlabel(labels[column])
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(labels[row])
            elif row != column:
                axis.set_yticklabels([])
            axis.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8, frameon=False)
    figure.suptitle("Full four-dimensional bounded posterior", y=0.995)
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "four_dimensional_posterior.png",
        dpi=600,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT / "four_dimensional_posterior.pdf", bbox_inches="tight"
    )
    figure.savefig(
        OUTPUT / "four_dimensional_posterior.svg", bbox_inches="tight"
    )
    plt.close(figure)


def coverage_plot(mle_rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    colors = {"internal_basis": "#2b8cbe", "external_references": "#e34a33"}
    for protocol in colors:
        rows = [
            row
            for row in mle_rows
            if row["protocol"] == protocol and row["parameter"] == "temperature"
        ]
        axes[0].plot(
            [row["total_binary_readouts"] for row in rows],
            [row["rmse"] for row in rows],
            "o-",
            color=colors[protocol],
            label=protocol.replace("_", " ").capitalize(),
        )
        axes[0].plot(
            [row["total_binary_readouts"] for row in rows],
            [row["predicted_sd"] for row in rows],
            "--",
            color=colors[protocol],
            alpha=0.7,
        )
        x_values = np.asarray(
            [row["total_binary_readouts"] for row in rows]
        )
        y_values = np.asarray([row["wald_95_coverage"] for row in rows])
        intervals = np.asarray(
            [row["wald_coverage_monte_carlo_95_interval"] for row in rows]
        )
        axes[1].errorbar(
            x_values,
            y_values,
            yerr=np.vstack([y_values - intervals[:, 0], intervals[:, 1] - y_values]),
            fmt="o-",
            color=colors[protocol],
            capsize=2.5,
            label=protocol.replace("_", " ").capitalize(),
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("total binary readouts")
    axes[0].set_ylabel("temperature RMSE / Fisher SD")
    axes[0].set_title("Finite-sample precision")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.75, 1.01)
    axes[1].set_xlabel("total binary readouts")
    axes[1].set_ylabel("nominal 95% Wald coverage")
    axes[1].set_title("Coverage with Monte Carlo uncertainty")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "finite_sample_coverage.png", dpi=600, bbox_inches="tight"
    )
    figure.savefig(OUTPUT / "finite_sample_coverage.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "finite_sample_coverage.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    designs = build_designs()
    mle_rows, mle_estimates = run_mle_ensemble(designs)
    coverage = {}
    for protocol_index, (protocol, design) in enumerate(designs.items()):
        coverage[protocol] = {}
        for total_readouts in TOTAL_READOUTS:
            coverage[protocol][str(total_readouts)] = covariance_coverage(
                design,
                allocations(protocol, total_readouts),
                seed=171000 + 1000 * protocol_index + int(np.log10(total_readouts)),
            )

    posterior_records = []
    posterior_samples: dict[str, np.ndarray] = {}
    posterior_counts: dict[str, np.ndarray] = {}
    for protocol_index, (protocol, design) in enumerate(designs.items()):
        record, samples, counts = run_posterior_case(
            protocol,
            design,
            total_readouts=8_000_000,
            seed=191000 + protocol_index * 1000,
        )
        posterior_records.append(record)
        posterior_samples[protocol] = samples
        posterior_counts[protocol] = counts

    with (OUTPUT / "mle_ensemble.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mle_rows[0]))
        writer.writeheader()
        writer.writerows(mle_rows)
    np.savez_compressed(
        OUTPUT / "four_dimensional_inference_data.npz",
        true_vector=TRUE_VECTOR,
        **{f"mle_{key}": value for key, value in mle_estimates.items()},
        **{f"posterior_{key}": value for key, value in posterior_samples.items()},
        **{f"counts_{key}": value for key, value in posterior_counts.items()},
    )

    temperature_rows = [
        row for row in mle_rows if row["parameter"] == "temperature"
    ]
    summary = {
        "parameters": dict(zip(PARAMETER_NAMES, TRUE_VECTOR.tolist(), strict=True)),
        "record_length": LENGTH,
        "independent_unit": (
            "one independently prepared length-8 trajectory block; each "
            "ground/excited reference is one independent binary readout"
        ),
        "protocols": {
            "internal_basis": {
                "all_z_block_fraction": 1.0 - INTERNAL_X_FRACTION,
                "all_x_block_fraction": INTERNAL_X_FRACTION,
            },
            "external_references": {
                "sensing_readout_fraction": EXTERNAL_FRACTIONS[0],
                "ground_reference_fraction": EXTERNAL_FRACTIONS[1],
                "excited_reference_fraction": EXTERNAL_FRACTIONS[2],
            },
        },
        "mle_replicates_per_case": MLE_REPLICATES,
        "mle_temperature_results": temperature_rows,
        "linearized_coverage_diagnostics": coverage,
        "posterior_cases": posterior_records,
        "posterior_diagnostic_thresholds": {
            "split_rhat": 1.05,
            "bulk_ess": 200,
        },
        "interval_definition": "equal-tailed 95% credible intervals; Wald intervals use estimate +/- 1.96 SE",
        "interpretation": (
            "Full four-dimensional inference confirms that internal basis "
            "self-calibration remains strongly correlated and non-Gaussian at "
            "finite budget, while explicit reference data regularize all four "
            "parameters and make Fisher/Wald approximations reliable much "
            "earlier."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pair_plot(posterior_samples, posterior_records)
    coverage_plot(mle_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
