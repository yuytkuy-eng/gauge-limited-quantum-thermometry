"""Four-dimensional coverage grid and Bayesian prior-sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


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
    MCMCResult,
    bulk_effective_sample_size,
    gauge_log_absolute_jacobian,
    gauge_to_physical_coordinates,
    importance_reweighted_summary,
    log_posterior,
    physical_to_gauge_coordinates,
    physical_to_gauge_jacobian,
    run_random_walk_metropolis,
    split_rhat,
)


OUTPUT = ROOT / "results" / "parameter_grid_robustness"
LENGTH = 8
TOTAL_READOUTS = 8_000_000
SYSTEM_MEMORY_ANGLE = 0.55
INTERNAL_X_FRACTION = 0.325
EXTERNAL_FRACTIONS = (0.795, 0.185, 0.020)
BOUNDS = np.asarray(
    [[0.30, 2.00], [0.02, 1.35], [0.001, 0.15], [0.001, 0.35]]
)
PRIOR = BoundedUniformPrior(BOUNDS)
PARAMETER_NAMES = ("temperature", "memory_angle", "false_positive", "false_negative")
PARAMETER_SYMBOLS = ("T", "mu", "alpha", "beta")
GRID_AXES = {
    "temperature": np.asarray([0.6, 0.9, 1.3]),
    "memory_angle": np.asarray([0.2, 0.5, 0.9]),
    "false_positive": np.asarray([0.005, 0.02, 0.06]),
    "false_negative": np.asarray([0.01, 0.04, 0.12]),
}
LINEAR_REPLICATES = 2000
EXACT_MLE_REPLICATES = 24
POSTERIOR_DRAWS = 4000
POSTERIOR_BURN_IN = 1600
STRESS_POINTS = {
    "central": np.asarray([0.9, 0.5, 0.02, 0.04]),
    "cold_clean_high_memory": np.asarray([0.6, 0.9, 0.005, 0.01]),
    "hot_noisy_low_memory": np.asarray([1.3, 0.2, 0.06, 0.12]),
}
PRIOR_NAMES = (
    "uniform_wide",
    "log_temperature",
    "low_error_beta_1_10",
    "jeffreys_assignment",
    "restricted_support",
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def binomial_wilson_interval(successes: int, trials: int) -> tuple[float, float]:
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


def allocations(protocol: str) -> tuple[int, ...]:
    if protocol == "internal_basis":
        total_blocks = TOTAL_READOUTS // LENGTH
        x_blocks = int(round(INTERNAL_X_FRACTION * total_blocks))
        return total_blocks - x_blocks, x_blocks
    if protocol == "external_references":
        sensing_blocks = int(
            round(TOTAL_READOUTS * EXTERNAL_FRACTIONS[0] / LENGTH)
        )
        ground_shots = int(round(TOTAL_READOUTS * EXTERNAL_FRACTIONS[1]))
        excited_shots = TOTAL_READOUTS - LENGTH * sensing_blocks - ground_shots
        return sensing_blocks, ground_shots, excited_shots
    raise ValueError(f"unknown protocol: {protocol}")


def information(probability: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    return (jacobian / probability) @ jacobian.T


def build_design(vector: np.ndarray, protocol: str) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    template = CollisionParameters(
        temperature=float(vector[0]),
        memory_angle=float(vector[1]),
        system_memory_angle=SYSTEM_MEMORY_ANGLE,
    )
    z_probability, z_jacobian = quantum_probability_jacobian(
        vector,
        template,
        0.5 * np.pi,
        LENGTH,
        measurement_polar_angles=0.0,
    )
    if protocol == "internal_basis":
        x_probability, x_jacobian = quantum_probability_jacobian(
            vector,
            template,
            0.5 * np.pi,
            LENGTH,
            measurement_polar_angles=0.5 * np.pi,
        )
        return (
            ("z", z_probability, z_jacobian),
            ("x", x_probability, x_jacobian),
        )
    ground_probability, ground_jacobian = calibration_probability_jacobian(
        vector, 0
    )
    excited_probability, excited_jacobian = calibration_probability_jacobian(
        vector, 1
    )
    return (
        ("z", z_probability, z_jacobian),
        ("ground", ground_probability, ground_jacobian),
        ("excited", excited_probability, excited_jacobian),
    )


def fisher_for_design(
    design: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    allocation: tuple[int, ...],
) -> np.ndarray:
    fisher = np.zeros((4, 4))
    for (_, probability, jacobian), count in zip(
        design, allocation, strict=True
    ):
        fisher += count * information(probability, jacobian)
    return 0.5 * (fisher + fisher.T)


def simulate_counts(
    design: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    allocation: tuple[int, ...],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    return [
        rng.multinomial(count, probability)
        for (_, probability, _), count in zip(design, allocation, strict=True)
    ]


def score_estimate(
    truth: np.ndarray,
    design: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    allocation: tuple[int, ...],
    counts: list[np.ndarray],
    covariance: np.ndarray,
) -> np.ndarray:
    score = np.zeros(4)
    for (_, probability, jacobian), count, observed in zip(
        design, allocation, counts, strict=True
    ):
        score += jacobian @ ((observed - count * probability) / probability)
    estimate = np.clip(
        truth + covariance @ score, BOUNDS[:, 0], BOUNDS[:, 1]
    )
    if estimate[2] + estimate[3] >= PRIOR.maximum_assignment_sum:
        return truth.copy()
    return estimate


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


def run_linearized_coverage(
    truth: np.ndarray,
    design: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    allocation: tuple[int, ...],
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    score = np.zeros((LINEAR_REPLICATES, 4))
    for (_, probability, jacobian), count in zip(
        design, allocation, strict=True
    ):
        counts = rng.multinomial(count, probability, size=LINEAR_REPLICATES)
        score += (counts - count * probability) @ (jacobian / probability).T
    estimates = truth + score @ covariance.T
    standard_deviation = np.sqrt(np.diag(covariance))
    coverage = np.abs(estimates - truth) <= 1.96 * standard_deviation
    physical = (
        np.all(estimates >= BOUNDS[:, 0], axis=1)
        & np.all(estimates <= BOUNDS[:, 1], axis=1)
        & (estimates[:, 2] + estimates[:, 3] < PRIOR.maximum_assignment_sum)
    )
    return {
        "coverage": np.mean(coverage, axis=0),
        "bias": np.mean(estimates - truth, axis=0),
        "empirical_sd": np.std(estimates, axis=0, ddof=1),
        "physical_fraction": float(np.mean(physical)),
        "negative_assignment_fraction": float(
            np.mean((estimates[:, 2] < 0.0) | (estimates[:, 3] < 0.0))
        ),
    }


def run_exact_mle_coverage(
    truth: np.ndarray,
    protocol: str,
    design: tuple[tuple[str, np.ndarray, np.ndarray], ...],
    allocation: tuple[int, ...],
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray | float], np.ndarray]:
    template = CollisionParameters(
        temperature=float(truth[0]),
        memory_angle=float(truth[1]),
        system_memory_angle=SYSTEM_MEMORY_ANGLE,
    )
    estimates = np.empty((EXACT_MLE_REPLICATES, 4))
    standard_errors = np.empty_like(estimates)
    converged = np.zeros(EXACT_MLE_REPLICATES, dtype=bool)
    boundary = np.zeros_like(estimates, dtype=bool)
    iterations = np.empty(EXACT_MLE_REPLICATES, dtype=int)
    for replicate in range(EXACT_MLE_REPLICATES):
        counts = simulate_counts(design, allocation, rng)
        fit = fisher_scoring_mle(
            experiments_from_counts(protocol, counts),
            template,
            LENGTH,
            score_estimate(truth, design, allocation, counts, covariance),
            bounds=BOUNDS,
            max_iterations=24,
        )
        estimates[replicate] = fit.estimate
        standard_errors[replicate] = np.sqrt(
            np.maximum(np.diag(fit.covariance), 0.0)
        )
        converged[replicate] = fit.converged
        iterations[replicate] = fit.iterations
        boundary[replicate] = (
            (fit.estimate - BOUNDS[:, 0] < 2e-4)
            | (BOUNDS[:, 1] - fit.estimate < 2e-4)
        )
    errors = estimates - truth
    covered = np.abs(errors) <= 1.96 * standard_errors
    return (
        {
            "coverage": np.mean(covered, axis=0),
            "coverage_successes": np.count_nonzero(covered, axis=0),
            "bias": np.mean(errors, axis=0),
            "empirical_sd": np.std(estimates, axis=0, ddof=1),
            "rmse": np.sqrt(np.mean(errors**2, axis=0)),
            "mean_reported_se": np.mean(standard_errors, axis=0),
            "converged_fraction": float(np.mean(converged)),
            "boundary_fraction": np.mean(boundary, axis=0),
            "mean_iterations": float(np.mean(iterations)),
        },
        estimates,
    )


def grid_row(
    grid_index: int, protocol: str, truth: np.ndarray
) -> tuple[dict[str, float | int | str], np.ndarray]:
    design = build_design(truth, protocol)
    allocation = allocations(protocol)
    fisher = fisher_for_design(design, allocation)
    covariance = np.linalg.inv(fisher)
    fisher_sd = np.sqrt(np.diag(covariance))
    protocol_index = 0 if protocol == "internal_basis" else 1
    linear = run_linearized_coverage(
        truth,
        design,
        allocation,
        covariance,
        np.random.default_rng(240000 + 10000 * protocol_index + grid_index),
    )
    exact, estimates = run_exact_mle_coverage(
        truth,
        protocol,
        design,
        allocation,
        covariance,
        np.random.default_rng(340000 + 10000 * protocol_index + grid_index),
    )
    eigenvalues = np.linalg.eigvalsh(fisher)
    row: dict[str, float | int | str] = {
        "grid_index": grid_index,
        "protocol": protocol,
        "total_binary_readouts": TOTAL_READOUTS,
        "independent_units": int(sum(allocation)),
        "temperature": float(truth[0]),
        "memory_angle": float(truth[1]),
        "false_positive": float(truth[2]),
        "false_negative": float(truth[3]),
        "fisher_rank": int(np.linalg.matrix_rank(fisher, tol=1e-8)),
        "fisher_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "linear_replicates": LINEAR_REPLICATES,
        "linear_physical_fraction": float(linear["physical_fraction"]),
        "linear_negative_assignment_fraction": float(
            linear["negative_assignment_fraction"]
        ),
        "exact_mle_replicates": EXACT_MLE_REPLICATES,
        "mle_converged_fraction": float(exact["converged_fraction"]),
        "mle_mean_iterations": float(exact["mean_iterations"]),
    }
    for parameter, symbol in enumerate(PARAMETER_SYMBOLS):
        lower, upper = binomial_wilson_interval(
            int(exact["coverage_successes"][parameter]),
            EXACT_MLE_REPLICATES,
        )
        row.update(
            {
                f"fisher_sd_{symbol}": float(fisher_sd[parameter]),
                f"linear_coverage_{symbol}": float(
                    linear["coverage"][parameter]
                ),
                f"linear_bias_{symbol}": float(linear["bias"][parameter]),
                f"linear_empirical_sd_{symbol}": float(
                    linear["empirical_sd"][parameter]
                ),
                f"mle_coverage_{symbol}": float(exact["coverage"][parameter]),
                f"mle_coverage_wilson_low_{symbol}": lower,
                f"mle_coverage_wilson_high_{symbol}": upper,
                f"mle_bias_{symbol}": float(exact["bias"][parameter]),
                f"mle_empirical_sd_{symbol}": float(
                    exact["empirical_sd"][parameter]
                ),
                f"mle_rmse_{symbol}": float(exact["rmse"][parameter]),
                f"mle_mean_reported_se_{symbol}": float(
                    exact["mean_reported_se"][parameter]
                ),
                f"mle_boundary_fraction_{symbol}": float(
                    exact["boundary_fraction"][parameter]
                ),
            }
        )
    return row, estimates


def run_grid_point(
    task: tuple[int, np.ndarray]
) -> tuple[int, list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    """Run both protocols for one grid point in an independent process."""

    grid_index, truth = task
    rows: list[dict[str, float | int | str]] = []
    estimates: dict[str, np.ndarray] = {}
    for protocol in ("internal_basis", "external_references"):
        row, values = grid_row(grid_index, protocol, truth)
        rows.append(row)
        estimates[f"grid_{grid_index}_{protocol}"] = values
    return grid_index, rows, estimates


def positive_definite_covariance(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-14)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def posterior_initials(
    center: np.ndarray, covariance: np.ndarray, seed: int
) -> np.ndarray:
    covariance = positive_definite_covariance(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(4):
        for _ in range(2000):
            proposal = center + 0.65 * root @ rng.normal(size=4)
            if PRIOR.contains(proposal):
                starts.append(proposal)
                break
        else:
            starts.append(center.copy())
    return np.asarray(starts)


def run_blocked_gauge_metropolis(
    log_density,
    initials: np.ndarray,
    covariance: np.ndarray,
    *,
    draws: int,
    burn_in: int,
    seed: int,
) -> tuple[MCMCResult, np.ndarray, np.ndarray]:
    """Blocked sampler with a global temperature move along the gauge ridge.

    The temperature proposal is independent uniform on the physical bounds,
    while ``(mu, alpha, kappa)`` receives a conditional Gaussian random-walk
    move. Both proposals are symmetric, so no proposal-density term is needed.
    """

    starts = np.asarray(initials, dtype=float)
    gauge_covariance = positive_definite_covariance(covariance)
    nuisance_covariance = (
        gauge_covariance[1:, 1:]
        - np.outer(gauge_covariance[1:, 0], gauge_covariance[0, 1:])
        / gauge_covariance[0, 0]
    )
    nuisance_covariance = positive_definite_covariance(nuisance_covariance)
    ridge_slope = gauge_covariance[1:, 0] / gauge_covariance[0, 0]
    eigenvalues, eigenvectors = np.linalg.eigh(nuisance_covariance)
    nuisance_root = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(seed)
    chain_count = starts.shape[0]
    samples = np.empty((chain_count, draws, 4))
    log_values = np.empty((chain_count, draws))
    temperature_acceptance = np.empty(chain_count)
    nuisance_acceptance = np.empty(chain_count)
    scales = np.ones(chain_count)
    temperature_scales = np.full(chain_count, 0.12)
    total_iterations = burn_in + draws
    for chain in range(chain_count):
        current = starts[chain].copy()
        current_log = float(log_density(current))
        accepted_temperature = 0
        accepted_nuisance = 0
        accepted_nuisance_window = 0
        accepted_temperature_window = 0
        for iteration in range(total_iterations):
            temperature_proposal = current.copy()
            temperature_proposal[0] = current[0] + (
                temperature_scales[chain] * rng.normal()
            )
            temperature_proposal[1:] += ridge_slope * (
                temperature_proposal[0] - current[0]
            )
            proposal_log = float(log_density(temperature_proposal))
            if (
                np.isfinite(proposal_log)
                and np.log(rng.random()) < proposal_log - current_log
            ):
                current = temperature_proposal
                current_log = proposal_log
                accepted_temperature += 1
                accepted_temperature_window += 1

            nuisance_proposal = current.copy()
            nuisance_proposal[1:] += scales[chain] * (
                nuisance_root @ rng.normal(size=3)
            )
            proposal_log = float(log_density(nuisance_proposal))
            if (
                np.isfinite(proposal_log)
                and np.log(rng.random()) < proposal_log - current_log
            ):
                current = nuisance_proposal
                current_log = proposal_log
                accepted_nuisance += 1
                accepted_nuisance_window += 1
            if iteration < burn_in and (iteration + 1) % 50 == 0:
                rate = accepted_nuisance_window / 50.0
                scales[chain] *= float(
                    np.exp(np.clip(rate - 0.32, -0.25, 0.25))
                )
                scales[chain] = float(np.clip(scales[chain], 0.05, 20.0))
                temperature_rate = accepted_temperature_window / 50.0
                temperature_scales[chain] *= float(
                    np.exp(np.clip(temperature_rate - 0.30, -0.25, 0.25))
                )
                temperature_scales[chain] = float(
                    np.clip(temperature_scales[chain], 0.005, 0.50)
                )
                accepted_nuisance_window = 0
                accepted_temperature_window = 0
            if iteration >= burn_in:
                saved = iteration - burn_in
                samples[chain, saved] = current
                log_values[chain, saved] = current_log
        temperature_acceptance[chain] = accepted_temperature / total_iterations
        nuisance_acceptance[chain] = accepted_nuisance / total_iterations
    result = MCMCResult(
        samples=samples,
        log_posterior=log_values,
        acceptance_rate=0.5 * (
            temperature_acceptance + nuisance_acceptance
        ),
        split_rhat=split_rhat(samples),
        bulk_effective_sample_size=bulk_effective_sample_size(samples),
        proposal_scales=scales,
    )
    return (
        result,
        temperature_acceptance,
        nuisance_acceptance,
        temperature_scales,
    )


def run_independence_gauge_metropolis(
    log_density,
    *,
    ridge_temperature_nodes: np.ndarray,
    ridge_nuisance_nodes: np.ndarray,
    conditional_covariance: np.ndarray,
    draws: int,
    burn_in: int,
    seed: int,
) -> tuple[MCMCResult, np.ndarray]:
    """Independence Metropolis sampler for a broad, prior-truncated ridge.

    Temperature is proposed uniformly over the full support. The three
    nuisance gauge coordinates are proposed from a Gaussian centered on the
    Fisher ridge, truncated by physical bounds. Because the Gaussian mean is
    deterministic at each temperature, the unnormalized proposal density is
    available exactly; the common truncation normalizer cancels from the
    independence Hastings ratio.
    """

    rng = np.random.default_rng(seed)
    chain_count = 4
    samples = np.empty((chain_count, draws, 4))
    log_values = np.empty((chain_count, draws))
    acceptance = np.empty(chain_count)
    total_iterations = burn_in + draws
    covariance = positive_definite_covariance(conditional_covariance)
    inverse_covariance = np.linalg.inv(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def proposal() -> tuple[np.ndarray, float]:
        temperature = rng.uniform(BOUNDS[0, 0], BOUNDS[0, 1])
        mean = np.asarray(
            [
                np.interp(
                    temperature,
                    ridge_temperature_nodes,
                    ridge_nuisance_nodes[:, parameter],
                )
                for parameter in range(3)
            ]
        )
        nuisance = mean + root @ rng.normal(size=3)
        vector = np.concatenate(([temperature], nuisance))
        residual = nuisance - mean
        return vector, float(-0.5 * residual @ inverse_covariance @ residual)

    for chain in range(chain_count):
        for _ in range(100000):
            current, current_log_q = proposal()
            current_log = float(log_density(current))
            if np.isfinite(current_log):
                break
        else:
            raise RuntimeError("failed to initialize independence sampler")
        accepted = 0
        for iteration in range(total_iterations):
            candidate, candidate_log_q = proposal()
            candidate_log = float(log_density(candidate))
            log_ratio = (
                candidate_log
                - current_log
                + current_log_q
                - candidate_log_q
            )
            if np.isfinite(candidate_log) and np.log(rng.random()) < log_ratio:
                current = candidate
                current_log = candidate_log
                current_log_q = candidate_log_q
                accepted += 1
            if iteration >= burn_in:
                saved = iteration - burn_in
                samples[chain, saved] = current
                log_values[chain, saved] = current_log
        acceptance[chain] = accepted / total_iterations
    result = MCMCResult(
        samples=samples,
        log_posterior=log_values,
        acceptance_rate=acceptance,
        split_rhat=split_rhat(samples),
        bulk_effective_sample_size=bulk_effective_sample_size(samples),
        proposal_scales=np.ones(chain_count),
    )
    return result, acceptance


def fit_profile_gauge_ridge(
    log_density,
    center: np.ndarray,
    conditional_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile the gauge ridge on deterministic temperature nodes."""

    temperature_nodes = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], 25)
    if not np.all(np.diff(temperature_nodes) > 0.0):
        raise RuntimeError("profile temperature nodes must be strictly increasing")
    nuisance_nodes = np.empty((temperature_nodes.size, 3))
    center_physical = gauge_to_physical_coordinates(center)
    start = np.asarray(center_physical[1:], dtype=float)
    inverse_scale = np.diag(
        1.0 / np.sqrt(np.maximum(np.diag(conditional_covariance), 1e-14))
    )
    for index, temperature in enumerate(temperature_nodes):
        def objective(physical_nuisance: np.ndarray) -> float:
            physical = np.concatenate(([temperature], physical_nuisance))
            if not PRIOR.contains(physical):
                return 1e100
            value = float(
                log_density(physical_to_gauge_coordinates(physical))
            )
            if not np.isfinite(value):
                return 1e100
            return -value

        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            bounds=[tuple(BOUNDS[index]) for index in range(1, 4)],
            options={"maxiter": 500, "xatol": 2e-7, "fatol": 2e-4},
        )
        if not np.isfinite(result.fun) or result.fun >= 1e99:
            raise RuntimeError("profile-ridge optimization failed")
        start = result.x
        nuisance_nodes[index] = physical_to_gauge_coordinates(
            np.concatenate(([temperature], start))
        )[1:]
    differences = np.diff(nuisance_nodes, axis=0)
    scaled_step = np.max(np.abs((inverse_scale @ differences.T).T))
    if scaled_step > 100.0:
        raise RuntimeError("profile-ridge optimization is discontinuous")
    return temperature_nodes, nuisance_nodes


def alternative_log_prior(samples: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    temperature = values[:, 0]
    alpha = values[:, 2]
    beta = values[:, 3]
    if name == "uniform_wide":
        return np.zeros(values.shape[0])
    if name == "log_temperature":
        return -np.log(temperature)
    if name == "low_error_beta_1_10":
        return 9.0 * (np.log1p(-alpha) + np.log1p(-beta))
    if name == "jeffreys_assignment":
        return -0.5 * (
            np.log(alpha)
            + np.log1p(-alpha)
            + np.log(beta)
            + np.log1p(-beta)
        )
    if name == "restricted_support":
        contained = (
            (values[:, 0] >= 0.45)
            & (values[:, 0] <= 1.55)
            & (values[:, 1] >= 0.10)
            & (values[:, 1] <= 1.10)
            & (values[:, 2] <= 0.10)
            & (values[:, 3] <= 0.20)
        )
        result = np.full(values.shape[0], -np.inf)
        result[contained] = 0.0
        return result
    raise ValueError(f"unknown prior: {name}")


def run_baseline_posterior(
    point_name: str,
    truth: np.ndarray,
    protocol: str,
    seed: int,
) -> tuple[list[dict[str, float | int | str | bool]], np.ndarray, dict[str, object]]:
    design = build_design(truth, protocol)
    allocation = allocations(protocol)
    fisher = fisher_for_design(design, allocation)
    covariance = positive_definite_covariance(np.linalg.inv(fisher))
    rng = np.random.default_rng(seed)
    counts = simulate_counts(design, allocation, rng)
    experiments = experiments_from_counts(protocol, counts)
    template = CollisionParameters(
        temperature=float(truth[0]),
        memory_angle=float(truth[1]),
        system_memory_angle=SYSTEM_MEMORY_ANGLE,
    )
    initial = score_estimate(truth, design, allocation, counts, covariance)
    fit = fisher_scoring_mle(
        experiments,
        template,
        LENGTH,
        initial,
        bounds=BOUNDS,
        max_iterations=30,
    )
    center = fit.estimate if PRIOR.contains(fit.estimate) else truth
    if protocol == "internal_basis":
        physical_initials = posterior_initials(center, covariance, seed + 1)
        gauge_initials = np.asarray(
            [physical_to_gauge_coordinates(value) for value in physical_initials]
        )
        gauge_covariance = positive_definite_covariance(
            physical_to_gauge_jacobian(center)
            @ covariance
            @ physical_to_gauge_jacobian(center).T
        )

        def density(gauge_vector: np.ndarray) -> float:
            if gauge_vector[0] <= 0.0:
                return float("-inf")
            physical = gauge_to_physical_coordinates(gauge_vector)
            return log_posterior(
                physical, template, experiments, LENGTH, PRIOR
            ) + gauge_log_absolute_jacobian(gauge_vector)

        if point_name == "hot_noisy_low_memory":
            gauge_center = physical_to_gauge_coordinates(center)
            conditional_covariance = (
                gauge_covariance[1:, 1:]
                - np.outer(gauge_covariance[1:, 0], gauge_covariance[0, 1:])
                / gauge_covariance[0, 0]
            )
            ridge_temperature_nodes, ridge_nuisance_nodes = fit_profile_gauge_ridge(
                density, gauge_center, conditional_covariance
            )
            result, temperature_acceptance = run_independence_gauge_metropolis(
                density,
                ridge_temperature_nodes=ridge_temperature_nodes,
                ridge_nuisance_nodes=ridge_nuisance_nodes,
                conditional_covariance=conditional_covariance,
                draws=POSTERIOR_DRAWS,
                burn_in=POSTERIOR_BURN_IN,
                seed=seed + 2,
            )
            nuisance_acceptance = None
            temperature_scales = None
        else:
            result = run_random_walk_metropolis(
                density,
                gauge_initials,
                1.35 * gauge_covariance,
                draws=POSTERIOR_DRAWS,
                burn_in=POSTERIOR_BURN_IN,
                seed=seed + 2,
            )
            temperature_acceptance = None
            nuisance_acceptance = None
            temperature_scales = None
        samples = np.empty_like(result.samples)
        for chain in range(samples.shape[0]):
            for draw in range(samples.shape[1]):
                samples[chain, draw] = gauge_to_physical_coordinates(
                    result.samples[chain, draw]
                )
        coordinate_system = "gauge-adapted (T,mu,alpha,kappa)"
    else:
        initials = posterior_initials(center, covariance, seed + 1)
        result = run_random_walk_metropolis(
            lambda vector: log_posterior(
                vector, template, experiments, LENGTH, PRIOR
            ),
            initials,
            1.35 * covariance,
            draws=POSTERIOR_DRAWS,
            burn_in=POSTERIOR_BURN_IN,
            seed=seed + 2,
        )
        samples = result.samples
        coordinate_system = "physical (T,mu,alpha,beta)"
        temperature_acceptance = None
        nuisance_acceptance = None
        temperature_scales = None
    flattened = samples.reshape(-1, 4)
    baseline = importance_reweighted_summary(
        flattened, np.zeros(flattened.shape[0])
    ).summary
    rows: list[dict[str, float | int | str | bool]] = []
    for prior_name in PRIOR_NAMES:
        reweighted = importance_reweighted_summary(
            flattened, alternative_log_prior(flattened, prior_name)
        )
        summary = reweighted.summary
        rows.append(
            {
                "point": point_name,
                "protocol": protocol,
                "prior": prior_name,
                "temperature": float(truth[0]),
                "memory_angle": float(truth[1]),
                "false_positive": float(truth[2]),
                "false_negative": float(truth[3]),
                "posterior_mean_T": float(summary.mean[0]),
                "posterior_sd_T": float(summary.standard_deviation[0]),
                "posterior_ci_low_T": float(summary.interval_95[0, 0]),
                "posterior_ci_high_T": float(summary.interval_95[0, 1]),
                "true_T_covered": bool(
                    summary.interval_95[0, 0]
                    <= truth[0]
                    <= summary.interval_95[0, 1]
                ),
                "mean_shift_in_baseline_sd": float(
                    (summary.mean[0] - baseline.mean[0])
                    / baseline.standard_deviation[0]
                ),
                "sd_ratio_to_baseline": float(
                    summary.standard_deviation[0]
                    / baseline.standard_deviation[0]
                ),
                "importance_ess": reweighted.effective_sample_size,
                "importance_ess_fraction": reweighted.effective_sample_fraction,
                "maximum_normalized_weight": reweighted.maximum_normalized_weight,
            }
        )
    diagnostics = {
        "point": point_name,
        "protocol": protocol,
        "truth": truth.tolist(),
        "allocation": list(allocation),
        "mle": fit.estimate.tolist(),
        "mle_converged": bool(fit.converged),
        "coordinate_system": coordinate_system,
        "acceptance_rate": result.acceptance_rate.tolist(),
        "global_temperature_acceptance_rate": (
            None
            if temperature_acceptance is None
            else temperature_acceptance.tolist()
        ),
        "conditional_nuisance_acceptance_rate": (
            None if nuisance_acceptance is None else nuisance_acceptance.tolist()
        ),
        "global_temperature_proposal_scale": (
            None if temperature_scales is None else temperature_scales.tolist()
        ),
        "split_rhat_sampling_coordinates": result.split_rhat.tolist(),
        "bulk_ess_sampling_coordinates": result.bulk_effective_sample_size.tolist(),
        "draws_per_chain": POSTERIOR_DRAWS,
        "burn_in_per_chain": POSTERIOR_BURN_IN,
        "chains": 4,
    }
    return rows, samples, diagnostics


def matrix_from_rows(
    rows: list[dict[str, float | int | str]],
    protocol: str,
    row_axis: str,
    column_axis: str,
    value: str,
    aggregation: str,
) -> np.ndarray:
    row_values = GRID_AXES[row_axis]
    column_values = GRID_AXES[column_axis]
    matrix = np.empty((len(row_values), len(column_values)))
    for row_index, row_value in enumerate(row_values):
        for column_index, column_value in enumerate(column_values):
            selected = [
                float(item[value])
                for item in rows
                if item["protocol"] == protocol
                and np.isclose(float(item[row_axis]), row_value)
                and np.isclose(float(item[column_axis]), column_value)
            ]
            if aggregation == "mean":
                matrix[row_index, column_index] = float(np.mean(selected))
            elif aggregation == "min":
                matrix[row_index, column_index] = float(np.min(selected))
            elif aggregation == "median":
                matrix[row_index, column_index] = float(np.median(selected))
            elif aggregation == "max":
                matrix[row_index, column_index] = float(np.max(selected))
            else:
                raise ValueError("unknown aggregation")
    return matrix


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    threshold = 0.5 * (float(np.nanmin(matrix)) + float(np.nanmax(matrix)))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], fmt),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if matrix[row, column] < threshold else "black",
            )


def coverage_grid_figure(rows: list[dict[str, float | int | str]]) -> None:
    protocols = ("internal_basis", "external_references")
    figure, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.75), constrained_layout=True
    )
    coverage_images = []
    ratio_images = []
    for row_index, protocol in enumerate(protocols):
        temperature_memory_coverage = matrix_from_rows(
            rows,
            protocol,
            "memory_angle",
            "temperature",
            "mle_coverage_T",
            "mean",
        )
        detector_coverage = matrix_from_rows(
            rows,
            protocol,
            "false_positive",
            "false_negative",
            "mle_coverage_T",
            "mean",
        )
        ratios = []
        for item in rows:
            if item["protocol"] == protocol:
                copied = dict(item)
                copied["rmse_ratio"] = float(item["mle_rmse_T"]) / float(
                    item["fisher_sd_T"]
                )
                ratios.append(copied)
        rmse_ratio = matrix_from_rows(
            ratios,
            protocol,
            "memory_angle",
            "temperature",
            "rmse_ratio",
            "median",
        )
        image_a = axes[row_index, 0].imshow(
            temperature_memory_coverage,
            origin="lower",
            vmin=0.75,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
        )
        image_b = axes[row_index, 1].imshow(
            detector_coverage,
            origin="lower",
            vmin=0.75,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
        )
        image_c = axes[row_index, 2].imshow(
            rmse_ratio,
            origin="lower",
            vmin=0.65,
            vmax=2.0,
            cmap="magma",
            aspect="auto",
        )
        coverage_images.extend([image_a, image_b])
        ratio_images.append(image_c)
        annotate_heatmap(axes[row_index, 0], temperature_memory_coverage, ".2f")
        annotate_heatmap(axes[row_index, 1], detector_coverage, ".2f")
        annotate_heatmap(axes[row_index, 2], rmse_ratio, ".2f")
        axes[row_index, 0].set_xticks(range(3), GRID_AXES["temperature"])
        axes[row_index, 0].set_yticks(range(3), GRID_AXES["memory_angle"])
        axes[row_index, 0].set_xlabel(r"temperature $T$")
        axes[row_index, 0].set_ylabel(r"memory angle $\mu$")
        axes[row_index, 1].set_xticks(range(3), GRID_AXES["false_negative"])
        axes[row_index, 1].set_yticks(range(3), GRID_AXES["false_positive"])
        axes[row_index, 1].set_xlabel(r"false-negative $\beta$")
        axes[row_index, 1].set_ylabel(r"false-positive $\alpha$")
        axes[row_index, 2].set_xticks(range(3), GRID_AXES["temperature"])
        axes[row_index, 2].set_yticks(range(3), GRID_AXES["memory_angle"])
        axes[row_index, 2].set_xlabel(r"temperature $T$")
        axes[row_index, 2].set_ylabel(r"memory angle $\mu$")
        axes[row_index, 0].text(
            -0.52,
            0.5,
            "Internal basis" if row_index == 0 else "External references",
            rotation=90,
            rotation_mode="anchor",
            va="center",
            ha="center",
            transform=axes[row_index, 0].transAxes,
            fontsize=8,
            fontweight="bold",
        )
    axes[0, 0].set_title("MLE coverage\nmean over detector grid")
    axes[0, 1].set_title("MLE coverage\nmean over physical grid")
    axes[0, 2].set_title("RMSE / Fisher SD\nmedian over detector grid")
    figure.colorbar(
        coverage_images[0],
        ax=axes[:, :2].ravel().tolist(),
        label="nominal 95% temperature coverage",
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        shrink=0.75,
    )
    figure.colorbar(
        ratio_images[0],
        ax=axes[:, 2].ravel().tolist(),
        label="RMSE / Fisher SD",
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        shrink=0.88,
    )
    for column, label in enumerate(("a", "b", "c")):
        axes[0, column].text(
            -0.22,
            1.16,
            label,
            transform=axes[0, column].transAxes,
            fontweight="bold",
            fontsize=9,
        )
    figure.savefig(
        OUTPUT / "parameter_grid_coverage.png", dpi=600, bbox_inches="tight"
    )
    figure.savefig(OUTPUT / "parameter_grid_coverage.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "parameter_grid_coverage.svg", bbox_inches="tight")
    plt.close(figure)


def prior_sensitivity_figure(
    rows: list[dict[str, float | int | str | bool]]
) -> None:
    point_order = list(STRESS_POINTS)
    prior_order = list(PRIOR_NAMES)
    colors = {
        "uniform_wide": "#252525",
        "log_temperature": "#2b8cbe",
        "low_error_beta_1_10": "#41ab5d",
        "jeffreys_assignment": "#dd1c77",
        "restricted_support": "#f16913",
    }
    labels = {
        "uniform_wide": "uniform",
        "log_temperature": r"$p(T)\propto1/T$",
        "low_error_beta_1_10": "low-error",
        "jeffreys_assignment": "Jeffreys errors",
        "restricted_support": "restricted support",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for axis, protocol in zip(
        axes[0], ("internal_basis", "external_references"), strict=True
    ):
        for point_index, point in enumerate(point_order):
            truth = STRESS_POINTS[point][0]
            axis.axvline(truth, color="#bdbdbd", linewidth=0.7, zorder=0)
            for prior_index, prior_name in enumerate(prior_order):
                item = next(
                    row
                    for row in rows
                    if row["protocol"] == protocol
                    and row["point"] == point
                    and row["prior"] == prior_name
                )
                y = point_index + (prior_index - 2) * 0.115
                low = float(item["posterior_ci_low_T"])
                high = float(item["posterior_ci_high_T"])
                mean = float(item["posterior_mean_T"])
                axis.errorbar(
                    mean,
                    y,
                    xerr=np.asarray([[mean - low], [high - mean]]),
                    fmt="o",
                    color=colors[prior_name],
                    ms=3,
                    capsize=1.5,
                    linewidth=0.9,
                    label=labels[prior_name] if point_index == 0 else None,
                )
        axis.set_yticks(range(3), ["central", "cold, clean,\nhigh memory", "hot, noisy,\nlow memory"])
        axis.invert_yaxis()
        axis.set_xlabel(r"temperature posterior mean and 95% interval")
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(
            "Internal basis" if protocol == "internal_basis" else "External references"
        )
    axes[0, 0].legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.13),
        fontsize=6,
    )
    row_labels = [
        f"{'internal' if protocol == 'internal_basis' else 'external'}: {point.replace('_', ' ')}"
        for protocol in ("internal_basis", "external_references")
        for point in point_order
    ]
    shifts = np.empty((6, 4))
    ess = np.empty((6, 4))
    alternative_priors = prior_order[1:]
    for row_index, (protocol, point) in enumerate(
        itertools.product(
            ("internal_basis", "external_references"), point_order
        )
    ):
        for column_index, prior_name in enumerate(alternative_priors):
            item = next(
                row
                for row in rows
                if row["protocol"] == protocol
                and row["point"] == point
                and row["prior"] == prior_name
            )
            shifts[row_index, column_index] = float(
                item["mean_shift_in_baseline_sd"]
            )
            ess[row_index, column_index] = float(item["importance_ess_fraction"])
    limit = max(1.0, float(np.max(np.abs(shifts))))
    shift_image = axes[1, 0].imshow(
        shifts, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
    )
    ess_image = axes[1, 1].imshow(
        ess, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto"
    )
    short_prior_labels = [r"$1/T$", "low-error", "Jeffreys", "support"]
    for axis in axes[1]:
        axis.set_xticks(
            range(4),
            short_prior_labels,
            rotation=25,
            rotation_mode="anchor",
            ha="right",
        )
        axis.set_yticks(range(6), row_labels)
    annotate_heatmap(axes[1, 0], shifts, ".2f")
    annotate_heatmap(axes[1, 1], ess, ".2f")
    axes[1, 0].set_title("posterior-mean shift / baseline SD")
    axes[1, 1].set_title("importance effective-sample fraction")
    figure.colorbar(shift_image, ax=axes[1, 0], fraction=0.045, pad=0.035)
    figure.colorbar(ess_image, ax=axes[1, 1], fraction=0.045, pad=0.035)
    figure.text(0.008, 0.985, "a", fontweight="bold", fontsize=9)
    figure.text(0.507, 0.985, "b", fontweight="bold", fontsize=9)
    figure.text(0.008, 0.492, "c", fontweight="bold", fontsize=9)
    figure.text(0.507, 0.492, "d", fontweight="bold", fontsize=9)
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95), h_pad=2.2, w_pad=1.2)
    figure.savefig(OUTPUT / "prior_sensitivity.png", dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT / "prior_sensitivity.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "prior_sensitivity.svg", bbox_inches="tight")
    plt.close(figure)


def summarize_grid(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for protocol in ("internal_basis", "external_references"):
        selected = [row for row in rows if row["protocol"] == protocol]
        worst_coverage = min(selected, key=lambda row: float(row["mle_coverage_T"]))
        worst_ratio = max(
            selected,
            key=lambda row: float(row["mle_rmse_T"]) / float(row["fisher_sd_T"]),
        )
        minimum_convergence = min(
            selected, key=lambda row: float(row["mle_converged_fraction"])
        )
        pooled_successes = int(
            sum(
                round(
                    float(row["mle_coverage_T"])
                    * int(row["exact_mle_replicates"])
                )
                for row in selected
            )
        )
        pooled_trials = int(
            sum(int(row["exact_mle_replicates"]) for row in selected)
        )
        summary[protocol] = {
            "descriptive_pooled_exact_mle_temperature_coverage": (
                pooled_successes / pooled_trials
            ),
            "descriptive_pooled_coverage_wilson_95_interval": list(
                binomial_wilson_interval(pooled_successes, pooled_trials)
            ),
            "mean_linearized_temperature_coverage": float(
                np.mean([float(row["linear_coverage_T"]) for row in selected])
            ),
            "mean_mle_convergence_fraction": float(
                np.mean(
                    [float(row["mle_converged_fraction"]) for row in selected]
                )
            ),
            "cells_with_convergence_below_0_9": int(
                sum(
                    float(row["mle_converged_fraction"]) < 0.9
                    for row in selected
                )
            ),
            "cells_whose_wilson_interval_contains_0_95": int(
                sum(
                    float(row["mle_coverage_wilson_low_T"])
                    <= 0.95
                    <= float(row["mle_coverage_wilson_high_T"])
                    for row in selected
                )
            ),
            "linearized_temperature_coverage_range": [
                min(float(row["linear_coverage_T"]) for row in selected),
                max(float(row["linear_coverage_T"]) for row in selected),
            ],
            "exact_mle_temperature_coverage_range_per_24_replicate_cell": [
                min(float(row["mle_coverage_T"]) for row in selected),
                max(float(row["mle_coverage_T"]) for row in selected),
            ],
            "fisher_temperature_sd_range": [
                min(float(row["fisher_sd_T"]) for row in selected),
                max(float(row["fisher_sd_T"]) for row in selected),
            ],
            "rmse_to_fisher_sd_range": [
                min(
                    float(row["mle_rmse_T"]) / float(row["fisher_sd_T"])
                    for row in selected
                ),
                max(
                    float(row["mle_rmse_T"]) / float(row["fisher_sd_T"])
                    for row in selected
                ),
            ],
            "convergence_fraction_range": [
                min(float(row["mle_converged_fraction"]) for row in selected),
                max(float(row["mle_converged_fraction"]) for row in selected),
            ],
            "linear_physical_fraction_range": [
                min(float(row["linear_physical_fraction"]) for row in selected),
                max(float(row["linear_physical_fraction"]) for row in selected),
            ],
            "descriptive_worst_coverage_point": {
                key: worst_coverage[key]
                for key in (
                    "temperature",
                    "memory_angle",
                    "false_positive",
                    "false_negative",
                    "mle_coverage_T",
                    "mle_coverage_wilson_low_T",
                    "mle_coverage_wilson_high_T",
                )
            },
            "descriptive_largest_rmse_ratio_point": {
                "temperature": worst_ratio["temperature"],
                "memory_angle": worst_ratio["memory_angle"],
                "false_positive": worst_ratio["false_positive"],
                "false_negative": worst_ratio["false_negative"],
                "rmse_to_fisher_sd": float(worst_ratio["mle_rmse_T"])
                / float(worst_ratio["fisher_sd_T"]),
            },
            "descriptive_minimum_convergence_point": {
                key: minimum_convergence[key]
                for key in (
                    "temperature",
                    "memory_angle",
                    "false_positive",
                    "false_negative",
                    "mle_converged_fraction",
                )
            },
        }
    return summary


def summarize_priors(
    rows: list[dict[str, float | int | str | bool]],
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    all_alternatives = [
        row for row in rows if row["prior"] != "uniform_wide"
    ]
    by_case: dict[str, object] = {}
    for protocol in ("internal_basis", "external_references"):
        by_case[protocol] = {}
        for point in STRESS_POINTS:
            selected = [
                row
                for row in rows
                if row["protocol"] == protocol and row["point"] == point
            ]
            baseline = next(
                row for row in selected if row["prior"] == "uniform_wide"
            )
            case_alternatives = [
                row for row in selected if row["prior"] != "uniform_wide"
            ]
            by_case[protocol][point] = {
                "baseline_temperature_mean": float(
                    baseline["posterior_mean_T"]
                ),
                "baseline_temperature_sd": float(baseline["posterior_sd_T"]),
                "baseline_temperature_95_interval": [
                    float(baseline["posterior_ci_low_T"]),
                    float(baseline["posterior_ci_high_T"]),
                ],
                "maximum_absolute_mean_shift_in_baseline_sd": float(
                    max(
                        abs(float(row["mean_shift_in_baseline_sd"]))
                        for row in case_alternatives
                    )
                ),
                "alternative_sd_ratio_range": [
                    min(
                        float(row["sd_ratio_to_baseline"])
                        for row in case_alternatives
                    ),
                    max(
                        float(row["sd_ratio_to_baseline"])
                        for row in case_alternatives
                    ),
                ],
                "minimum_importance_ess_fraction": float(
                    min(
                        float(row["importance_ess_fraction"])
                        for row in case_alternatives
                    )
                ),
            }
    return {
        "maximum_absolute_temperature_mean_shift_in_baseline_sd": float(
            max(
                abs(float(row["mean_shift_in_baseline_sd"]))
                for row in all_alternatives
            )
        ),
        "temperature_sd_ratio_range": [
            min(
                float(row["sd_ratio_to_baseline"])
                for row in all_alternatives
            ),
            max(
                float(row["sd_ratio_to_baseline"])
                for row in all_alternatives
            ),
        ],
        "minimum_importance_ess_fraction": float(
            min(
                float(row["importance_ess_fraction"])
                for row in all_alternatives
            )
        ),
        "maximum_normalized_weight": float(
            max(
                float(row["maximum_normalized_weight"])
                for row in all_alternatives
            )
        ),
        "mcmc_maximum_split_rhat": float(
            max(
                max(record["split_rhat_sampling_coordinates"])
                for record in diagnostics
            )
        ),
        "mcmc_minimum_bulk_ess": float(
            min(
                min(record["bulk_ess_sampling_coordinates"])
                for record in diagnostics
            )
        ),
        "by_protocol_and_stress_point": by_case,
    }


def all_grid_points() -> list[np.ndarray]:
    """Return the deterministic lexicographic 3 x 3 x 3 x 3 grid."""

    return [
        np.asarray(values, dtype=float)
        for values in itertools.product(
            GRID_AXES["temperature"],
            GRID_AXES["memory_angle"],
            GRID_AXES["false_positive"],
            GRID_AXES["false_negative"],
        )
    ]


def run_grid_range(start: int, stop: int) -> None:
    """Calculate and persist an independently reproducible grid shard."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid_points = all_grid_points()
    if start < 0 or stop > len(grid_points) or start >= stop:
        raise ValueError("grid shard must satisfy 0 <= start < stop <= 81")
    rows: list[dict[str, float | int | str]] = []
    estimates: dict[str, list[list[float]]] = {}
    for grid_index in range(start, stop):
        _, point_rows, point_estimates = run_grid_point(
            (grid_index, grid_points[grid_index])
        )
        rows.extend(point_rows)
        estimates.update(
            {key: value.tolist() for key, value in point_estimates.items()}
        )
        print(
            f"completed shard grid point {grid_index + 1}/81",
            flush=True,
        )
    payload = {"start": start, "stop": stop, "rows": rows, "estimates": estimates}
    (OUTPUT / f"grid_shard_{start:02d}_{stop:02d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def load_grid_shards() -> tuple[
    list[dict[str, float | int | str]], dict[str, np.ndarray]
]:
    """Load nonoverlapping grid shards and verify all 81 points are present."""

    rows: list[dict[str, float | int | str]] = []
    estimates: dict[str, np.ndarray] = {}
    for path in sorted(OUTPUT.glob("grid_shard_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        estimates.update(
            {
                key: np.asarray(value, dtype=float)
                for key, value in payload["estimates"].items()
            }
        )
    keys = {(int(row["grid_index"]), str(row["protocol"])) for row in rows}
    expected = {
        (index, protocol)
        for index in range(81)
        for protocol in ("internal_basis", "external_references")
    }
    if keys != expected or len(rows) != len(expected):
        missing = sorted(expected - keys)
        duplicated = len(rows) - len(keys)
        raise RuntimeError(
            f"grid shards incomplete: {len(missing)} missing cells, "
            f"{duplicated} duplicate rows"
        )
    rows.sort(key=lambda row: (int(row["grid_index"]), str(row["protocol"])))
    return rows, estimates


def prior_cases() -> list[tuple[str, np.ndarray, str]]:
    """Return the deterministic three-point by two-protocol posterior cases."""

    return [
        (point_name, truth, protocol)
        for point_name, truth in STRESS_POINTS.items()
        for protocol in ("internal_basis", "external_references")
    ]


def run_prior_case_index(case_index: int) -> None:
    """Calculate and checkpoint one baseline posterior and all reweightings."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = prior_cases()
    if case_index < 0 or case_index >= len(cases):
        raise ValueError("prior case index must lie in [0, 6)")
    point_name, truth, protocol = cases[case_index]
    rows, samples, diagnostics = run_baseline_posterior(
        point_name,
        truth,
        protocol,
        seed=440000 + 1000 * case_index,
    )
    (OUTPUT / f"prior_case_{case_index}.json").write_text(
        json.dumps({"rows": rows, "diagnostics": diagnostics}),
        encoding="utf-8",
    )
    np.savez_compressed(
        OUTPUT / f"prior_case_{case_index}.npz", samples=samples
    )
    print(
        f"completed prior case {case_index + 1}/6: {point_name}, {protocol}",
        flush=True,
    )


def load_prior_cases() -> tuple[
    list[dict[str, float | int | str | bool]],
    dict[str, np.ndarray],
    list[dict[str, object]],
]:
    """Load all six independently checkpointed posterior cases."""

    rows: list[dict[str, float | int | str | bool]] = []
    samples: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, object]] = []
    for case_index, (point_name, _, protocol) in enumerate(prior_cases()):
        json_path = OUTPUT / f"prior_case_{case_index}.json"
        npz_path = OUTPUT / f"prior_case_{case_index}.npz"
        if not json_path.exists() or not npz_path.exists():
            raise RuntimeError(f"missing checkpoint for prior case {case_index}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        diagnostics.append(payload["diagnostics"])
        with np.load(npz_path) as archive:
            samples[f"posterior_{point_name}_{protocol}"] = archive[
                "samples"
            ]
    return rows, samples, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-shard",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="run only grid indices in the half-open interval [START, STOP)",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="assemble complete grid shards and run prior-sensitivity posteriors",
    )
    parser.add_argument(
        "--prior-case",
        type=int,
        help="run and checkpoint one posterior case indexed from 0 to 5",
    )
    args = parser.parse_args()
    if args.grid_shard is not None:
        run_grid_range(*args.grid_shard)
        return
    if args.prior_case is not None:
        run_prior_case_index(args.prior_case)
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid_points = all_grid_points()
    if args.assemble:
        grid_rows, exact_estimates = load_grid_shards()
    else:
        grid_rows = []
        exact_estimates = {}
        for grid_index, truth in enumerate(grid_points):
            _, rows, estimates = run_grid_point((grid_index, truth))
            grid_rows.extend(rows)
            exact_estimates.update(estimates)
            if (grid_index + 1) % 9 == 0:
                print(
                    f"completed coverage grid points {grid_index + 1}/81",
                    flush=True,
                )

    if args.assemble:
        prior_rows, posterior_samples, diagnostics = load_prior_cases()
    else:
        prior_rows = []
        posterior_samples = {}
        diagnostics = []
        for case_index, (point_name, truth, protocol) in enumerate(prior_cases()):
            case_rows, samples, case_diagnostics = run_baseline_posterior(
                point_name,
                truth,
                protocol,
                seed=440000 + 1000 * case_index,
            )
            prior_rows.extend(case_rows)
            posterior_samples[f"posterior_{point_name}_{protocol}"] = samples
            diagnostics.append(case_diagnostics)
            print(
                f"completed prior case {case_index + 1}/6: {point_name}, {protocol}",
                flush=True,
            )

    with (OUTPUT / "parameter_grid_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    with (OUTPUT / "prior_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prior_rows[0]))
        writer.writeheader()
        writer.writerows(prior_rows)
    np.savez_compressed(
        OUTPUT / "parameter_grid_robustness_data.npz",
        grid_points=np.asarray(grid_points),
        **exact_estimates,
        **posterior_samples,
    )
    coverage_grid_figure(grid_rows)
    prior_sensitivity_figure(prior_rows)
    summary = {
        "grid": {
            "axes": {name: values.tolist() for name, values in GRID_AXES.items()},
            "points": len(grid_points),
            "protocols": 2,
            "total_binary_readouts_per_dataset": TOTAL_READOUTS,
            "linearized_replicates_per_point_and_protocol": LINEAR_REPLICATES,
            "exact_mle_replicates_per_point_and_protocol": EXACT_MLE_REPLICATES,
            "fixed_design_fractions": {
                "internal_x_block_fraction": INTERNAL_X_FRACTION,
                "external_sensing_ground_excited_readout_fractions": list(
                    EXTERNAL_FRACTIONS
                ),
            },
            "results": summarize_grid(grid_rows),
        },
        "prior_sensitivity": {
            "stress_points": {
                name: value.tolist() for name, value in STRESS_POINTS.items()
            },
            "baseline_prior": "uniform on the wide bounded physical domain",
            "alternative_priors": {
                "log_temperature": "p(T) proportional to 1/T",
                "low_error_beta_1_10": "independent Beta(1,10) kernels for alpha and beta",
                "jeffreys_assignment": "independent Beta(1/2,1/2) kernels for alpha and beta",
                "restricted_support": "T=[0.45,1.55], mu=[0.10,1.10], alpha<=0.10, beta<=0.20",
            },
            "importance_method": (
                "self-normalized alternative-to-baseline prior ratios on draws "
                "from the same exact-likelihood baseline posterior"
            ),
            "mcmc_diagnostics": diagnostics,
            "results": summarize_priors(prior_rows, diagnostics),
        },
        "independent_unit": (
            "one independently prepared length-8 trajectory block; every "
            "ground/excited reference shot is one independent binary unit"
        ),
        "coverage_interpretation": (
            "Per-cell exact-MLE coverage uses 24 datasets and is accompanied "
            "by Wilson intervals; grid extrema are descriptive stress-test "
            "statistics rather than multiplicity-adjusted hypothesis tests."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
