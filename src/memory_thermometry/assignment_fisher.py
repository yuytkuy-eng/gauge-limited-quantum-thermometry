"""Exact Fisher information with asymmetric readout errors as nuisances."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .classical_full_swap import classical_record_distribution
from .model import CollisionParameters, record_distribution as quantum_record_distribution


Array = np.ndarray


@dataclass(frozen=True)
class AssignmentFisherResult:
    """Four-parameter Fisher result ordered as ``(T, mu, alpha, beta)``."""

    matrix: Array
    effective_temperature_information: float
    eigenvalues: Array
    rank: int
    condition_number: float
    normalization_error: float


def assignment_calibration_fisher(
    false_positive: float,
    false_negative: float,
    prepared_state: int,
) -> Array:
    """Per-shot Fisher matrix from a known ground or excited reference.

    A prepared ground state calibrates ``alpha``; a prepared excited state
    calibrates ``beta``. The returned parameter order is ``(T, mu, alpha,
    beta)`` so it can be added directly to the sensing Fisher matrix.
    """

    if prepared_state not in (0, 1):
        raise ValueError("prepared_state must be 0 or 1")
    if not 0.0 < false_positive < 1.0:
        raise ValueError("false_positive must lie in (0, 1)")
    if not 0.0 < false_negative < 1.0:
        raise ValueError("false_negative must lie in (0, 1)")
    matrix = np.zeros((4, 4))
    if prepared_state == 0:
        matrix[2, 2] = 1.0 / (
            false_positive * (1.0 - false_positive)
        )
    else:
        matrix[3, 3] = 1.0 / (
            false_negative * (1.0 - false_negative)
        )
    return matrix


def effective_target_information(
    matrix: Array,
    target_index: int = 0,
    relative_tolerance: float = 1e-10,
) -> float:
    """Schur-complement information after projecting out all nuisances.

    A pseudoinverse is used so the result remains well-defined when nuisance
    scores are internally redundant. It is the squared norm of the target
    score after projection onto the orthogonal complement of their span.
    """

    information = np.asarray(matrix, dtype=float)
    if information.ndim != 2 or information.shape[0] != information.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 <= target_index < information.shape[0]:
        raise ValueError("target_index is out of range")
    nuisance = [index for index in range(information.shape[0]) if index != target_index]
    if not nuisance:
        return float(max(information[target_index, target_index], 0.0))
    nuisance_block = information[np.ix_(nuisance, nuisance)]
    cross = information[target_index, nuisance]
    inverse = np.linalg.pinv(nuisance_block, rcond=relative_tolerance)
    effective = information[target_index, target_index] - cross @ inverse @ cross
    scale = max(abs(information[target_index, target_index]), 1.0)
    if effective < 0.0 and abs(effective) <= 1e-10 * scale:
        effective = 0.0
    return float(max(effective, 0.0))


def _distribution(
    parameters: CollisionParameters,
    length: int,
    false_positive: float,
    false_negative: float,
) -> Array:
    return classical_record_distribution(
        parameters,
        length,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _physical_derivative(
    parameters: CollisionParameters,
    field: str,
    step: float,
    length: int,
    false_positive: float,
    false_negative: float,
) -> Array:
    value = float(getattr(parameters, field))
    lower = value - step
    upper = value + step
    if (field == "temperature" and lower <= 0.0) or (
        field == "memory_angle" and lower < 0.0
    ):
        base = _distribution(
            parameters, length, false_positive, false_negative
        )
        plus = _distribution(
            replace(parameters, **{field: upper}),
            length,
            false_positive,
            false_negative,
        )
        return (plus - base) / step
    if field == "memory_angle" and upper > 0.5 * np.pi:
        base = _distribution(
            parameters, length, false_positive, false_negative
        )
        minus = _distribution(
            replace(parameters, **{field: lower}),
            length,
            false_positive,
            false_negative,
        )
        return (base - minus) / step
    plus = _distribution(
        replace(parameters, **{field: upper}),
        length,
        false_positive,
        false_negative,
    )
    minus = _distribution(
        replace(parameters, **{field: lower}),
        length,
        false_positive,
        false_negative,
    )
    return (plus - minus) / (2.0 * step)


def _error_derivative(
    parameters: CollisionParameters,
    field: str,
    step: float,
    length: int,
    false_positive: float,
    false_negative: float,
) -> Array:
    alpha = float(false_positive)
    beta = float(false_negative)
    value = alpha if field == "false_positive" else beta
    lower = value - step
    upper = value + step
    if upper + (beta if field == "false_positive" else alpha) >= 1.0:
        raise ValueError("error derivative crosses alpha + beta = 1")
    if lower < 0.0:
        base = _distribution(parameters, length, alpha, beta)
        if field == "false_positive":
            plus = _distribution(parameters, length, upper, beta)
        else:
            plus = _distribution(parameters, length, alpha, upper)
        return (plus - base) / step
    if field == "false_positive":
        plus = _distribution(parameters, length, upper, beta)
        minus = _distribution(parameters, length, lower, beta)
    else:
        plus = _distribution(parameters, length, alpha, upper)
        minus = _distribution(parameters, length, alpha, lower)
    return (plus - minus) / (2.0 * step)


def assignment_fisher_matrix(
    parameters: CollisionParameters,
    length: int,
    false_positive: float,
    false_negative: float,
    temperature_step: float = 1e-4,
    memory_step: float = 1e-4,
    error_step: float = 1e-5,
    probability_floor: float = 1e-15,
    rank_tolerance: float = 1e-10,
) -> AssignmentFisherResult:
    """Exact record Fisher matrix for ``(T, mu, alpha, beta)``.

    The full-swap 2x2 instruments enumerate all ``2**length`` records. Physical
    and detector derivatives are evaluated by centered finite differences away
    from parameter boundaries.
    """

    if length < 1:
        raise ValueError("length must be at least 1")
    distribution = _distribution(
        parameters, length, false_positive, false_negative
    )
    derivatives = np.vstack(
        [
            _physical_derivative(
                parameters,
                "temperature",
                temperature_step,
                length,
                false_positive,
                false_negative,
            ),
            _physical_derivative(
                parameters,
                "memory_angle",
                memory_step,
                length,
                false_positive,
                false_negative,
            ),
            _error_derivative(
                parameters,
                "false_positive",
                error_step,
                length,
                false_positive,
                false_negative,
            ),
            _error_derivative(
                parameters,
                "false_negative",
                error_step,
                length,
                false_positive,
                false_negative,
            ),
        ]
    )
    mask = distribution > probability_floor
    selected = derivatives[:, mask]
    matrix = (selected / distribution[mask]) @ selected.T
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = float(max(eigenvalues[-1], 0.0))
    threshold = rank_tolerance * maximum
    positive = eigenvalues[eigenvalues > threshold]
    rank = int(positive.size)
    condition = (
        float(positive[-1] / positive[0])
        if rank == matrix.shape[0]
        else float("inf")
    )
    return AssignmentFisherResult(
        matrix=matrix,
        effective_temperature_information=effective_target_information(
            matrix, relative_tolerance=rank_tolerance
        ),
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition,
        normalization_error=float(abs(distribution.sum() - 1.0)),
    )


def quantum_assignment_fisher_matrix(
    parameters: CollisionParameters,
    probe_angles: float | Sequence[float],
    length: int,
    false_positive: float,
    false_negative: float,
    measurement_polar_angles: float | Sequence[float] = 0.0,
    measurement_azimuths: float | Sequence[float] = 0.0,
    temperature_step: float = 1e-4,
    memory_step: float = 1e-4,
    error_step: float = 1e-5,
    probability_floor: float = 1e-15,
    rank_tolerance: float = 1e-10,
) -> AssignmentFisherResult:
    """Exact four-parameter Fisher matrix for the general quantum trajectory.

    Unlike :func:`assignment_fisher_matrix`, this function permits arbitrary
    fixed or periodic probe-readout angles and retains the persistent
    thermometer state between cycles.  Known measurement-basis rotations can
    be supplied independently of the probe interaction angle.
    """

    if length < 1:
        raise ValueError("length must be at least 1")

    def distribution(
        candidate: CollisionParameters,
        alpha: float,
        beta: float,
    ) -> Array:
        return quantum_record_distribution(
            candidate,
            probe_angles,
            length,
            false_positive=alpha,
            false_negative=beta,
            measurement_polar_angles=measurement_polar_angles,
            measurement_azimuths=measurement_azimuths,
        )

    base = distribution(parameters, false_positive, false_negative)
    physical_derivatives = []
    for field, step in (
        ("temperature", temperature_step),
        ("memory_angle", memory_step),
    ):
        value = float(getattr(parameters, field))
        lower = value - step
        upper = value + step
        lower_invalid = (field == "temperature" and lower <= 0.0) or (
            field == "memory_angle" and lower < 0.0
        )
        upper_invalid = field == "memory_angle" and upper > 0.5 * np.pi
        if lower_invalid:
            plus = distribution(
                replace(parameters, **{field: upper}),
                false_positive,
                false_negative,
            )
            derivative = (plus - base) / step
        elif upper_invalid:
            minus = distribution(
                replace(parameters, **{field: lower}),
                false_positive,
                false_negative,
            )
            derivative = (base - minus) / step
        else:
            plus = distribution(
                replace(parameters, **{field: upper}),
                false_positive,
                false_negative,
            )
            minus = distribution(
                replace(parameters, **{field: lower}),
                false_positive,
                false_negative,
            )
            derivative = (plus - minus) / (2.0 * step)
        physical_derivatives.append(derivative)

    error_derivatives = []
    for field in ("false_positive", "false_negative"):
        alpha = float(false_positive)
        beta = float(false_negative)
        value = alpha if field == "false_positive" else beta
        lower = value - error_step
        upper = value + error_step
        other = beta if field == "false_positive" else alpha
        if upper + other >= 1.0:
            raise ValueError("error derivative crosses alpha + beta = 1")
        if field == "false_positive":
            plus = distribution(parameters, upper, beta)
            minus = (
                base
                if lower < 0.0
                else distribution(parameters, lower, beta)
            )
        else:
            plus = distribution(parameters, alpha, upper)
            minus = (
                base
                if lower < 0.0
                else distribution(parameters, alpha, lower)
            )
        denominator = error_step if lower < 0.0 else 2.0 * error_step
        error_derivatives.append((plus - minus) / denominator)

    derivatives = np.vstack([*physical_derivatives, *error_derivatives])
    mask = base > probability_floor
    selected = derivatives[:, mask]
    matrix = (selected / base[mask]) @ selected.T
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = float(max(eigenvalues[-1], 0.0))
    threshold = rank_tolerance * maximum
    positive = eigenvalues[eigenvalues > threshold]
    rank = int(positive.size)
    condition = (
        float(positive[-1] / positive[0])
        if rank == matrix.shape[0]
        else float("inf")
    )
    return AssignmentFisherResult(
        matrix=matrix,
        effective_temperature_information=effective_target_information(
            matrix, relative_tolerance=rank_tolerance
        ),
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition,
        normalization_error=float(abs(base.sum() - 1.0)),
    )
