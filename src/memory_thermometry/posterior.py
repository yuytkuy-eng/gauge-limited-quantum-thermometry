"""Bounded four-parameter posterior inference and MCMC diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .inference import (
    BlockExperiment,
    CalibrationExperiment,
    block_log_likelihood,
)
from .model import CollisionParameters
from .classical_full_swap import thermal_excitation_probability


Array = np.ndarray


@dataclass(frozen=True)
class BoundedUniformPrior:
    """Uniform physical prior for ``(T,mu,alpha,beta)``.

    The detector prior is uniform over the rectangle specified by ``bounds``,
    intersected with ``alpha + beta < maximum_assignment_sum``.
    """

    bounds: Array
    maximum_assignment_sum: float = 0.95

    def validate(self) -> None:
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (4, 2):
            raise ValueError("bounds must have shape (4, 2)")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each lower bound must be below its upper bound")
        if bounds[0, 0] <= 0.0:
            raise ValueError("temperature lower bound must be positive")
        if bounds[1, 0] < 0.0 or bounds[1, 1] > 0.5 * np.pi:
            raise ValueError("memory-angle bounds must lie in [0, pi/2]")
        if np.any(bounds[2:, 0] < 0.0):
            raise ValueError("assignment-error bounds must be nonnegative")
        if not 0.0 < self.maximum_assignment_sum < 1.0:
            raise ValueError("maximum_assignment_sum must lie in (0, 1)")

    def contains(self, vector: Array) -> bool:
        self.validate()
        values = np.asarray(vector, dtype=float)
        bounds = np.asarray(self.bounds, dtype=float)
        return bool(
            values.shape == (4,)
            and np.all(values >= bounds[:, 0])
            and np.all(values <= bounds[:, 1])
            and values[2] + values[3] < self.maximum_assignment_sum
        )

    def log_density(self, vector: Array) -> float:
        return 0.0 if self.contains(vector) else float("-inf")


@dataclass(frozen=True)
class MCMCResult:
    """Posterior draws and convergence diagnostics from random-walk chains."""

    samples: Array
    log_posterior: Array
    acceptance_rate: Array
    split_rhat: Array
    bulk_effective_sample_size: Array
    proposal_scales: Array


@dataclass(frozen=True)
class PosteriorSummary:
    """Marginal summaries ordered as ``(T,mu,alpha,beta)``."""

    mean: Array
    median: Array
    standard_deviation: Array
    interval_95: Array
    correlation: Array


@dataclass(frozen=True)
class ImportanceReweightedSummary:
    """Posterior summary after stable self-normalized importance reweighting."""

    summary: PosteriorSummary
    effective_sample_size: float
    effective_sample_fraction: float
    maximum_normalized_weight: float


def physical_to_gauge_coordinates(
    vector: Array, energy_gap: float = 1.0
) -> Array:
    """Map ``(T,mu,alpha,beta)`` to ``(T,mu,alpha,kappa)``."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (4,):
        raise ValueError("parameter vector must have shape (4,)")
    excitation = thermal_excitation_probability(values[0], energy_gap)
    return np.asarray(
        [
            values[0],
            values[1],
            values[2],
            (1.0 - values[2] - values[3]) * excitation,
        ]
    )


def gauge_to_physical_coordinates(
    vector: Array, energy_gap: float = 1.0
) -> Array:
    """Map ``(T,mu,alpha,kappa)`` to ``(T,mu,alpha,beta)``."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (4,):
        raise ValueError("gauge vector must have shape (4,)")
    excitation = thermal_excitation_probability(values[0], energy_gap)
    return np.asarray(
        [
            values[0],
            values[1],
            values[2],
            1.0 - values[2] - values[3] / excitation,
        ]
    )


def gauge_log_absolute_jacobian(
    vector: Array, energy_gap: float = 1.0
) -> float:
    """Log ``|d(T,mu,alpha,beta)/d(T,mu,alpha,kappa)|``."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (4,):
        raise ValueError("gauge vector must have shape (4,)")
    if values[0] <= 0.0:
        return float("inf")
    excitation = thermal_excitation_probability(values[0], energy_gap)
    return float(-np.log(excitation))


def physical_to_gauge_jacobian(
    vector: Array, energy_gap: float = 1.0
) -> Array:
    """Jacobian of :func:`physical_to_gauge_coordinates`."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (4,):
        raise ValueError("parameter vector must have shape (4,)")
    temperature, _, alpha, beta = values
    excitation = thermal_excitation_probability(temperature, energy_gap)
    derivative = (
        energy_gap
        * excitation
        * (1.0 - excitation)
        / temperature**2
    )
    contrast = 1.0 - alpha - beta
    jacobian = np.eye(4)
    jacobian[3] = [contrast * derivative, 0.0, -excitation, -excitation]
    return jacobian


def log_posterior(
    vector: Array,
    template: CollisionParameters,
    experiments: Sequence[BlockExperiment | CalibrationExperiment],
    length: int,
    prior: BoundedUniformPrior,
) -> float:
    """Exact block-count log posterior up to an additive constant."""

    log_prior = prior.log_density(vector)
    if not np.isfinite(log_prior):
        return float("-inf")
    return log_prior + block_log_likelihood(
        vector, template, experiments, length
    )


def split_rhat(chains: Array) -> Array:
    """Rank-unadjusted split-Rhat for ``(chain, draw, parameter)`` samples."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("chains must have shape (>=2, >=4, parameters)")
    half = values.shape[1] // 2
    split = np.concatenate(
        [values[:, :half], values[:, -half:]], axis=0
    )
    draws = split.shape[1]
    within_variances = np.var(split, axis=1, ddof=1)
    within = np.mean(within_variances, axis=0)
    means = np.mean(split, axis=1)
    between = draws * np.var(means, axis=0, ddof=1)
    variance = ((draws - 1.0) / draws) * within + between / draws
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt(variance / within)
    result[(within == 0.0) & (between == 0.0)] = 1.0
    return result


def bulk_effective_sample_size(chains: Array) -> Array:
    """Conservative initial-positive-sequence autocorrelation ESS."""

    values = np.asarray(chains, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 4:
        raise ValueError("chains must have shape (chains, draws, parameters)")
    chain_count, draw_count, parameter_count = values.shape
    ess = np.empty(parameter_count)
    for parameter in range(parameter_count):
        centered = values[:, :, parameter] - np.mean(
            values[:, :, parameter], axis=1, keepdims=True
        )
        variance = float(np.mean(centered**2))
        if variance <= 0.0:
            ess[parameter] = chain_count * draw_count
            continue
        autocorrelations = []
        for lag in range(1, draw_count):
            covariance = float(
                np.mean(centered[:, :-lag] * centered[:, lag:])
            )
            autocorrelations.append(covariance / variance)
        autocorrelations = np.asarray(autocorrelations)
        positive_sum = 0.0
        for index in range(0, autocorrelations.size - 1, 2):
            pair = autocorrelations[index] + autocorrelations[index + 1]
            if pair <= 0.0:
                break
            positive_sum += pair
        tau = max(1.0 + 2.0 * positive_sum, 1.0)
        ess[parameter] = min(
            chain_count * draw_count / tau,
            chain_count * draw_count,
        )
    return ess


def posterior_summary(samples: Array) -> PosteriorSummary:
    """Return means, equal-tailed intervals, and posterior correlations."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("samples must end in four parameters")
    return PosteriorSummary(
        mean=np.mean(values, axis=0),
        median=np.median(values, axis=0),
        standard_deviation=np.std(values, axis=0, ddof=1),
        interval_95=np.quantile(values, [0.025, 0.975], axis=0).T,
        correlation=np.corrcoef(values, rowvar=False),
    )


def normalized_importance_weights(log_weights: Array) -> Array:
    """Normalize log importance ratios using a stable log-sum-exp shift."""

    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("log_weights must not be empty")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("at least one log weight must be finite")
    maximum = float(np.max(values[finite]))
    weights = np.zeros(values.size, dtype=float)
    weights[finite] = np.exp(values[finite] - maximum)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("importance weights cannot be normalized")
    return weights / total


def weighted_quantile(
    values: Array, quantiles: Array, weights: Array
) -> Array:
    """Return one-dimensional weighted quantiles by CDF interpolation."""

    observations = np.asarray(values, dtype=float).reshape(-1)
    probabilities = np.asarray(quantiles, dtype=float)
    normalized = np.asarray(weights, dtype=float).reshape(-1)
    if observations.size == 0 or observations.size != normalized.size:
        raise ValueError("values and weights must have the same nonzero size")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    if np.any(normalized < 0.0) or not np.isfinite(normalized).all():
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(normalized))
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    normalized = normalized / total
    order = np.argsort(observations)
    sorted_values = observations[order]
    sorted_weights = normalized[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative = np.concatenate(([0.0], cumulative, [1.0]))
    padded_values = np.concatenate(
        ([sorted_values[0]], sorted_values, [sorted_values[-1]])
    )
    return np.interp(probabilities, cumulative, padded_values)


def importance_reweighted_summary(
    samples: Array, log_weights: Array
) -> ImportanceReweightedSummary:
    """Summarize physical posterior draws under an alternative prior.

    ``log_weights`` is the pointwise alternative-to-baseline log-prior ratio.
    The likelihood cancels because all samples must come from the same
    baseline posterior.
    """

    values = np.asarray(samples, dtype=float)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("samples must end in four parameters")
    log_ratio = np.asarray(log_weights, dtype=float).reshape(-1)
    if log_ratio.size != values.shape[0]:
        raise ValueError("one log weight is required per posterior draw")
    weights = normalized_importance_weights(log_ratio)
    mean = weights @ values
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(standard_deviation, standard_deviation)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator
    correlation[~np.isfinite(correlation)] = 0.0
    np.fill_diagonal(correlation, 1.0)
    median = np.empty(4)
    interval = np.empty((4, 2))
    for parameter in range(4):
        median[parameter] = weighted_quantile(
            values[:, parameter], np.asarray([0.5]), weights
        )[0]
        interval[parameter] = weighted_quantile(
            values[:, parameter], np.asarray([0.025, 0.975]), weights
        )
    effective_sample_size = float(1.0 / np.sum(weights**2))
    summary = PosteriorSummary(
        mean=mean,
        median=median,
        standard_deviation=standard_deviation,
        interval_95=interval,
        correlation=correlation,
    )
    return ImportanceReweightedSummary(
        summary=summary,
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / values.shape[0],
        maximum_normalized_weight=float(np.max(weights)),
    )


def run_random_walk_metropolis(
    log_density: Callable[[Array], float],
    initials: Array,
    proposal_covariance: Array,
    *,
    draws: int,
    burn_in: int,
    thin: int = 1,
    seed: int = 0,
    adapt_interval: int = 50,
    target_acceptance: float = 0.28,
) -> MCMCResult:
    """Run multiple adaptive Gaussian random-walk Metropolis chains.

    Proposal scales are adapted only during burn-in and then frozen. The
    returned draws therefore come from a fixed-kernel Markov chain.
    """

    starts = np.asarray(initials, dtype=float)
    covariance = np.asarray(proposal_covariance, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 4:
        raise ValueError("initials must have shape (chains, 4)")
    if covariance.shape != (4, 4):
        raise ValueError("proposal_covariance must have shape (4, 4)")
    if draws < 1 or burn_in < 0 or thin < 1:
        raise ValueError("draws/thin must be positive and burn_in nonnegative")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-16)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, floor)))
    rng = np.random.default_rng(seed)
    chain_count = starts.shape[0]
    samples = np.empty((chain_count, draws, 4))
    log_values = np.empty((chain_count, draws))
    acceptance_rates = np.empty(chain_count)
    scales = np.ones(chain_count)
    total_iterations = burn_in + draws * thin
    for chain in range(chain_count):
        current = starts[chain].copy()
        current_log = float(log_density(current))
        if not np.isfinite(current_log):
            raise ValueError("every initial point must have finite log density")
        accepted_total = 0
        accepted_window = 0
        saved = 0
        for iteration in range(total_iterations):
            proposal = current + scales[chain] * (
                root @ rng.normal(size=4)
            )
            proposal_log = float(log_density(proposal))
            accepted = bool(
                np.isfinite(proposal_log)
                and np.log(rng.random()) < proposal_log - current_log
            )
            if accepted:
                current = proposal
                current_log = proposal_log
                accepted_total += 1
                accepted_window += 1
            if (
                iteration < burn_in
                and adapt_interval > 0
                and (iteration + 1) % adapt_interval == 0
            ):
                rate = accepted_window / adapt_interval
                scales[chain] *= float(
                    np.exp(np.clip(rate - target_acceptance, -0.25, 0.25))
                )
                scales[chain] = float(np.clip(scales[chain], 0.05, 20.0))
                accepted_window = 0
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples[chain, saved] = current
                log_values[chain, saved] = current_log
                saved += 1
        acceptance_rates[chain] = accepted_total / total_iterations
    return MCMCResult(
        samples=samples,
        log_posterior=log_values,
        acceptance_rate=acceptance_rates,
        split_rhat=split_rhat(samples),
        bulk_effective_sample_size=bulk_effective_sample_size(samples),
        proposal_scales=scales,
    )
