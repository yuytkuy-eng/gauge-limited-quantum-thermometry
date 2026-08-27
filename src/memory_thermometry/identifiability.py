"""Linear-algebra tools for gauge and local-identifiability calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PowerLawFit:
    """Log-log power-law fit ``value = coefficient * control**exponent``."""

    exponent: float
    coefficient: float
    r_squared: float


def whitened_probability_jacobian(
    probabilities: Array,
    jacobian: Array,
    probability_floor: float = 1e-15,
) -> Array:
    """Return score vectors in Euclidean coordinates.

    ``jacobian`` is ordered as ``(parameters, outcomes)``. Its rows divided by
    ``sqrt(probabilities)`` have Gram matrix equal to the classical Fisher
    matrix.
    """

    probability = np.asarray(probabilities, dtype=float)
    derivative = np.asarray(jacobian, dtype=float)
    if probability.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if derivative.ndim != 2 or derivative.shape[1] != probability.size:
        raise ValueError("jacobian must have shape (parameters, outcomes)")
    mask = probability > probability_floor
    if not np.any(mask):
        raise ValueError("no probabilities exceed probability_floor")
    return derivative[:, mask] / np.sqrt(probability[mask])


def efficient_score_residual(
    probabilities: Array,
    jacobian: Array,
    target_index: int = 0,
    relative_tolerance: float = 1e-10,
) -> Array:
    """Whitened target score after orthogonal nuisance projection."""

    whitened = whitened_probability_jacobian(probabilities, jacobian)
    if not 0 <= target_index < whitened.shape[0]:
        raise ValueError("target_index is out of range")
    nuisance_indices = [
        index for index in range(whitened.shape[0]) if index != target_index
    ]
    target = whitened[target_index]
    if not nuisance_indices:
        return target.copy()
    nuisance = whitened[nuisance_indices].T
    projection = nuisance @ np.linalg.pinv(
        nuisance, rcond=relative_tolerance
    )
    return target - projection @ target


def fit_power_law(
    controls: Sequence[float],
    values: Sequence[float],
) -> PowerLawFit:
    """Fit a positive power law by ordinary least squares in log coordinates."""

    control = np.asarray(controls, dtype=float)
    value = np.asarray(values, dtype=float)
    if control.ndim != 1 or value.ndim != 1 or control.size != value.size:
        raise ValueError("controls and values must be equal-length vectors")
    if control.size < 3:
        raise ValueError("at least three points are required")
    if np.any(control <= 0.0) or np.any(value <= 0.0):
        raise ValueError("power-law data must be strictly positive")
    log_control = np.log(control)
    log_value = np.log(value)
    exponent, intercept = np.polyfit(log_control, log_value, 1)
    fitted = intercept + exponent * log_control
    residual = float(np.sum((log_value - fitted) ** 2))
    total = float(np.sum((log_value - log_value.mean()) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - residual / total
    return PowerLawFit(
        exponent=float(exponent),
        coefficient=float(np.exp(intercept)),
        r_squared=float(r_squared),
    )


def similarity_tangent(
    instruments: Sequence[Array],
    initial: Array,
    final: Array,
    generator: Array,
) -> tuple[tuple[Array, ...], Array, Array]:
    """Return the infinitesimal tangent to a realization similarity orbit."""

    matrices = tuple(np.asarray(matrix) for matrix in instruments)
    right = np.asarray(initial)
    left = np.asarray(final)
    hamiltonian = np.asarray(generator)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("generator must be square")
    dimension = hamiltonian.shape[0]
    if right.shape != (dimension,) or left.shape != (dimension,):
        raise ValueError("initial and final must match the generator dimension")
    if any(matrix.shape != (dimension, dimension) for matrix in matrices):
        raise ValueError("all instruments must match the generator dimension")
    tangent = tuple(
        hamiltonian @ matrix - matrix @ hamiltonian for matrix in matrices
    )
    return tangent, hamiltonian @ right, -(left @ hamiltonian)


def record_probability_tangent(
    instruments: Sequence[Array],
    initial: Array,
    final: Array,
    record: Sequence[int],
    instrument_tangent: Sequence[Array],
    initial_tangent: Array,
    final_tangent: Array,
) -> tuple[float, float]:
    """Evaluate one word probability and its directional derivative."""

    state = np.asarray(initial)
    state_tangent = np.asarray(initial_tangent)
    matrices = tuple(np.asarray(matrix) for matrix in instruments)
    tangents = tuple(np.asarray(matrix) for matrix in instrument_tangent)
    if len(matrices) != len(tangents):
        raise ValueError("instrument and tangent alphabets must have equal size")
    for outcome in record:
        if not 0 <= int(outcome) < len(matrices):
            raise ValueError("record contains an invalid outcome")
        matrix = matrices[int(outcome)]
        tangent = tangents[int(outcome)]
        state_tangent = tangent @ state + matrix @ state_tangent
        state = matrix @ state
    probability = np.asarray(final) @ state
    derivative = np.asarray(final_tangent) @ state + np.asarray(final) @ state_tangent
    return float(np.real_if_close(probability)), float(np.real_if_close(derivative))

