"""Exact finite-record classical Fisher information calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .model import CollisionParameters, record_distribution


Array = np.ndarray


@dataclass(frozen=True)
class FisherResult:
    matrix: Array
    effective_temperature_information: float
    determinant: float
    condition_number: float
    normalization_error: float


def _parameter_derivative(
    parameters: CollisionParameters,
    field: str,
    step: float,
    probe_angles: float | Sequence[float],
    length: int,
) -> Array:
    value = float(getattr(parameters, field))
    lower = value - step
    upper = value + step

    if field == "temperature" and lower <= 0.0:
        base = record_distribution(parameters, probe_angles, length)
        plus = record_distribution(
            replace(parameters, temperature=upper), probe_angles, length
        )
        return (plus - base) / step

    if field == "memory_angle" and lower < 0.0:
        base = record_distribution(parameters, probe_angles, length)
        plus = record_distribution(
            replace(parameters, memory_angle=upper), probe_angles, length
        )
        return (plus - base) / step

    if field == "memory_angle" and upper > 0.5 * np.pi:
        base = record_distribution(parameters, probe_angles, length)
        minus = record_distribution(
            replace(parameters, memory_angle=lower), probe_angles, length
        )
        return (base - minus) / step

    plus = record_distribution(replace(parameters, **{field: upper}), probe_angles, length)
    minus = record_distribution(replace(parameters, **{field: lower}), probe_angles, length)
    return (plus - minus) / (2.0 * step)


def classical_fisher_matrix(
    parameters: CollisionParameters,
    probe_angles: float | Sequence[float],
    length: int,
    temperature_step: float = 1e-4,
    memory_step: float = 1e-4,
    probability_floor: float = 1e-14,
) -> FisherResult:
    """Calculate the Fisher matrix for `(temperature, memory_angle)`.

    The calculation uses the exact distribution over all binary records and
    finite differences only for the two parameter derivatives.
    """

    distribution = record_distribution(parameters, probe_angles, length)
    dp_temperature = _parameter_derivative(
        parameters,
        "temperature",
        temperature_step,
        probe_angles,
        length,
    )
    dp_memory = _parameter_derivative(
        parameters,
        "memory_angle",
        memory_step,
        probe_angles,
        length,
    )

    mask = distribution > probability_floor
    inverse_probability = 1.0 / distribution[mask]
    derivatives = np.vstack(
        [dp_temperature[mask], dp_memory[mask]]
    )
    matrix = (derivatives * inverse_probability) @ derivatives.T
    matrix = 0.5 * (matrix + matrix.T)

    nuisance_information = matrix[1, 1]
    if nuisance_information > 1e-14:
        effective = matrix[0, 0] - matrix[0, 1] ** 2 / nuisance_information
    else:
        effective = matrix[0, 0]
    effective = float(max(effective, 0.0))

    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = eigenvalues[eigenvalues > 1e-13]
    if positive.size < 2:
        condition = float("inf")
    else:
        condition = float(positive.max() / positive.min())

    return FisherResult(
        matrix=matrix,
        effective_temperature_information=effective,
        determinant=float(np.linalg.det(matrix)),
        condition_number=condition,
        normalization_error=float(abs(distribution.sum() - 1.0)),
    )

