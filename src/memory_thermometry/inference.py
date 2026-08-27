"""Finite-sample likelihood inference for noisy quantum trajectory blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .model import (
    CollisionParameters,
    normalize_measurement_schedule,
    normalize_schedule,
    record_distribution,
    record_distribution_superoperator,
)
from .classical_full_swap import classical_record_distribution


Array = np.ndarray
Schedule = float | Sequence[float]


@dataclass(frozen=True)
class BlockExperiment:
    """Counts of independent binary record blocks under one probe schedule."""

    probe_angles: Schedule
    counts: Array
    measurement_polar_angles: Schedule = 0.0
    measurement_azimuths: Schedule = 0.0


@dataclass(frozen=True)
class CalibrationExperiment:
    """Binary counts from a known ground or excited detector reference."""

    prepared_state: int
    counts: Array


@dataclass(frozen=True)
class MLEFit:
    """Result of bounded Fisher-scoring likelihood maximization."""

    estimate: Array
    covariance: Array
    log_likelihood: float
    iterations: int
    converged: bool
    score_norm: float


def _vector_to_parameters(
    vector: Array, template: CollisionParameters
) -> tuple[CollisionParameters, float, float]:
    values = np.asarray(vector, dtype=float)
    if values.shape != (4,):
        raise ValueError("parameter vector must have shape (4,)")
    parameters = replace(
        template,
        temperature=float(values[0]),
        memory_angle=float(values[1]),
    )
    return parameters, float(values[2]), float(values[3])


def quantum_probability_jacobian(
    vector: Array,
    template: CollisionParameters,
    probe_angles: Schedule,
    length: int,
    steps: Array | None = None,
    measurement_polar_angles: Schedule = 0.0,
    measurement_azimuths: Schedule = 0.0,
) -> tuple[Array, Array]:
    """Record probabilities and finite-difference Jacobian for four parameters."""

    values = np.asarray(vector, dtype=float)
    if steps is None:
        steps = np.asarray([1e-4, 1e-4, 1e-5, 1e-5])
    steps = np.asarray(steps, dtype=float)
    if values.shape != (4,) or steps.shape != (4,):
        raise ValueError("vector and steps must have shape (4,)")
    parameters, alpha, beta = _vector_to_parameters(values, template)
    def probability(
        candidate: CollisionParameters,
        candidate_alpha: float,
        candidate_beta: float,
    ) -> Array:
        probe_schedule = normalize_schedule(probe_angles, length)
        polar_schedule, azimuth_schedule = normalize_measurement_schedule(
            measurement_polar_angles, measurement_azimuths, length
        )
        if (
            np.allclose(probe_schedule, 0.5 * np.pi, atol=1e-14)
            and np.allclose(polar_schedule, 0.0, atol=1e-14)
            and np.allclose(azimuth_schedule, 0.0, atol=1e-14)
        ):
            return classical_record_distribution(
                candidate,
                length,
                false_positive=candidate_alpha,
                false_negative=candidate_beta,
            )
        return record_distribution_superoperator(
            candidate,
            probe_angles,
            length,
            false_positive=candidate_alpha,
            false_negative=candidate_beta,
            measurement_polar_angles=measurement_polar_angles,
            measurement_azimuths=measurement_azimuths,
        )

    base = probability(parameters, alpha, beta)
    jacobian = np.empty((4, base.size))
    for index, step in enumerate(steps):
        lower = values.copy()
        upper = values.copy()
        lower[index] -= step
        upper[index] += step
        lower_valid = (
            lower[0] > 0.0
            and lower[1] >= 0.0
            and lower[2] >= 0.0
            and lower[3] >= 0.0
            and lower[2] + lower[3] < 1.0
        )
        upper_valid = (
            upper[1] <= 0.5 * np.pi
            and upper[2] >= 0.0
            and upper[3] >= 0.0
            and upper[2] + upper[3] < 1.0
        )
        if lower_valid:
            lower_parameters, lower_alpha, lower_beta = _vector_to_parameters(
                lower, template
            )
            lower_probability = probability(
                lower_parameters, lower_alpha, lower_beta
            )
        else:
            lower_probability = base
        if upper_valid:
            upper_parameters, upper_alpha, upper_beta = _vector_to_parameters(
                upper, template
            )
            upper_probability = probability(
                upper_parameters, upper_alpha, upper_beta
            )
        else:
            upper_probability = base
        denominator = step * float(
            int(bool(lower_valid)) + int(bool(upper_valid))
        )
        jacobian[index] = (
            upper_probability - lower_probability
        ) / denominator
    return base, jacobian


def calibration_probability_jacobian(
    vector: Array, prepared_state: int
) -> tuple[Array, Array]:
    """Reference-readout probabilities and their four-parameter Jacobian."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (4,):
        raise ValueError("parameter vector must have shape (4,)")
    if prepared_state not in (0, 1):
        raise ValueError("prepared_state must be 0 or 1")
    alpha = float(values[2])
    beta = float(values[3])
    jacobian = np.zeros((4, 2))
    if prepared_state == 0:
        probability = np.asarray([1.0 - alpha, alpha])
        jacobian[2] = [-1.0, 1.0]
    else:
        probability = np.asarray([beta, 1.0 - beta])
        jacobian[3] = [1.0, -1.0]
    return probability, jacobian


def block_log_likelihood(
    vector: Array,
    template: CollisionParameters,
    experiments: Sequence[BlockExperiment | CalibrationExperiment],
    length: int,
) -> float:
    """Multinomial log likelihood, omitting parameter-independent constants."""

    parameters, alpha, beta = _vector_to_parameters(vector, template)
    value = 0.0
    for experiment in experiments:
        if isinstance(experiment, BlockExperiment):
            probe_schedule = normalize_schedule(
                experiment.probe_angles, length
            )
            polar_schedule, azimuth_schedule = normalize_measurement_schedule(
                experiment.measurement_polar_angles,
                experiment.measurement_azimuths,
                length,
            )
            if (
                np.allclose(probe_schedule, 0.5 * np.pi, atol=1e-14)
                and np.allclose(polar_schedule, 0.0, atol=1e-14)
                and np.allclose(azimuth_schedule, 0.0, atol=1e-14)
            ):
                probability = classical_record_distribution(
                    parameters,
                    length,
                    false_positive=alpha,
                    false_negative=beta,
                )
            else:
                probability = record_distribution_superoperator(
                    parameters,
                    experiment.probe_angles,
                    length,
                    false_positive=alpha,
                    false_negative=beta,
                    measurement_polar_angles=(
                        experiment.measurement_polar_angles
                    ),
                    measurement_azimuths=experiment.measurement_azimuths,
                )
        elif isinstance(experiment, CalibrationExperiment):
            probability, _ = calibration_probability_jacobian(
                vector, experiment.prepared_state
            )
        else:
            raise TypeError("unsupported experiment type")
        counts = np.asarray(experiment.counts, dtype=float)
        if counts.shape != probability.shape:
            raise ValueError("experiment counts have the wrong record dimension")
        value += float(np.dot(counts, np.log(np.maximum(probability, 1e-300))))
    return value


def score_and_fisher(
    vector: Array,
    template: CollisionParameters,
    experiments: Sequence[BlockExperiment | CalibrationExperiment],
    length: int,
    steps: Array | None = None,
) -> tuple[float, Array, Array]:
    """Log likelihood, score, and expected Fisher for block-count data."""

    log_likelihood = 0.0
    score = np.zeros(4)
    fisher = np.zeros((4, 4))
    for experiment in experiments:
        if isinstance(experiment, BlockExperiment):
            probability, jacobian = quantum_probability_jacobian(
                vector,
                template,
                experiment.probe_angles,
                length,
                steps,
                experiment.measurement_polar_angles,
                experiment.measurement_azimuths,
            )
        elif isinstance(experiment, CalibrationExperiment):
            probability, jacobian = calibration_probability_jacobian(
                vector, experiment.prepared_state
            )
        else:
            raise TypeError("unsupported experiment type")
        counts = np.asarray(experiment.counts, dtype=float)
        if counts.shape != probability.shape:
            raise ValueError("experiment counts have the wrong record dimension")
        safe_probability = np.maximum(probability, 1e-300)
        log_likelihood += float(np.dot(counts, np.log(safe_probability)))
        score += jacobian @ (counts / safe_probability)
        block_count = float(counts.sum())
        fisher += block_count * (
            (jacobian / safe_probability) @ jacobian.T
        )
    fisher = 0.5 * (fisher + fisher.T)
    return log_likelihood, score, fisher


def fisher_scoring_mle(
    experiments: Sequence[BlockExperiment | CalibrationExperiment],
    template: CollisionParameters,
    length: int,
    initial: Array,
    bounds: Array | None = None,
    steps: Array | None = None,
    max_iterations: int = 12,
    parameter_tolerance: float = 2e-5,
    score_tolerance: float = 2e-3,
) -> MLEFit:
    """Maximize the exact block likelihood with bounded Fisher scoring."""

    if bounds is None:
        bounds = np.asarray(
            [[0.30, 2.00], [0.02, 1.35], [0.001, 0.15], [0.001, 0.35]]
        )
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")
    estimate = np.clip(np.asarray(initial, dtype=float), bounds[:, 0], bounds[:, 1])
    converged = False
    final_score = np.full(4, np.nan)
    final_fisher = np.full((4, 4), np.nan)
    log_likelihood = block_log_likelihood(
        estimate, template, experiments, length
    )
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        current_log_likelihood, score, fisher = score_and_fisher(
            estimate, template, experiments, length, steps
        )
        inverse = np.linalg.pinv(fisher, rcond=1e-12)
        update = inverse @ score
        maximum_update = np.asarray([0.25, 0.18, 0.03, 0.06])
        scale = max(float(np.max(np.abs(update) / maximum_update)), 1.0)
        update /= scale
        accepted = False
        trial_scale = 1.0
        candidate = estimate
        candidate_log_likelihood = current_log_likelihood
        while trial_scale >= 1.0 / 128.0:
            trial = np.clip(
                estimate + trial_scale * update,
                bounds[:, 0],
                bounds[:, 1],
            )
            if trial[2] + trial[3] >= 0.95:
                trial_scale *= 0.5
                continue
            trial_log_likelihood = block_log_likelihood(
                trial, template, experiments, length
            )
            if trial_log_likelihood >= current_log_likelihood:
                candidate = trial
                candidate_log_likelihood = trial_log_likelihood
                accepted = True
                break
            trial_scale *= 0.5
        estimate_change = float(np.max(np.abs(candidate - estimate)))
        estimate = candidate
        log_likelihood = candidate_log_likelihood
        final_score = score
        final_fisher = fisher
        scaled_score_norm = float(
            np.sqrt(max(score @ inverse @ score, 0.0))
        )
        if estimate_change < parameter_tolerance and scaled_score_norm < score_tolerance:
            converged = True
            break
        if not accepted:
            break
    final_log_likelihood, final_score, final_fisher = score_and_fisher(
        estimate, template, experiments, length, steps
    )
    final_inverse = np.linalg.pinv(final_fisher, rcond=1e-12)
    return MLEFit(
        estimate=estimate,
        covariance=final_inverse,
        log_likelihood=final_log_likelihood,
        iterations=iterations,
        converged=converged,
        score_norm=float(
            np.sqrt(max(final_score @ final_inverse @ final_score, 0.0))
        ),
    )
