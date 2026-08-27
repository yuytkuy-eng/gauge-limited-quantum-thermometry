"""Collision thermometry with a two-qubit moving memory window.

The persistent state is ``S x M1 x M2``. After the thermometer collides with
``M1`` and is read out, ``M1`` collides with ``M2``, then ``M2`` collides with a
fresh Gibbs qubit ``E``. The old ``M1`` is discarded and ``(M2,E)`` becomes the
new two-step memory window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence

import numpy as np

from .assignment_fisher import AssignmentFisherResult, effective_target_information
from .model import (
    normalize_measurement_schedule,
    normalize_schedule,
    partial_swap_xy,
    readout_kraus,
    thermal_state,
)


Array = np.ndarray


def _validate_assignment_errors(
    false_positive: float, false_negative: float
) -> None:
    if not 0.0 <= false_positive < 1.0:
        raise ValueError("false_positive must lie in [0, 1)")
    if not 0.0 <= false_negative < 1.0:
        raise ValueError("false_negative must lie in [0, 1)")
    if false_positive + false_negative >= 1.0:
        raise ValueError("false_positive + false_negative must be less than 1")


@dataclass(frozen=True)
class TwoStepCollisionParameters:
    """Physical parameters for the moving two-memory collision chain."""

    temperature: float
    memory_angle: float
    system_memory_angle: float = 0.55
    energy_gap: float = 1.0

    def validate(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.energy_gap <= 0.0:
            raise ValueError("energy_gap must be positive")
        for name, angle in (
            ("memory_angle", self.memory_angle),
            ("system_memory_angle", self.system_memory_angle),
        ):
            if not 0.0 <= angle <= 0.5 * np.pi:
                raise ValueError(f"{name} must lie in [0, pi/2]")


@lru_cache(maxsize=512)
def _embedded_partial_swap(
    angle: float,
    qubit_count: int,
    first_qubit: int,
    second_qubit: int,
) -> Array:
    """Embed a two-qubit partial swap using big-endian tensor ordering."""

    if qubit_count < 2:
        raise ValueError("qubit_count must be at least two")
    if not 0 <= first_qubit < qubit_count:
        raise ValueError("first_qubit is out of range")
    if not 0 <= second_qubit < qubit_count or second_qubit == first_qubit:
        raise ValueError("second_qubit is out of range or duplicated")
    local = partial_swap_xy(angle)
    dimension = 2**qubit_count
    embedded = np.zeros((dimension, dimension), dtype=complex)
    for column in range(dimension):
        bits = [
            (column >> (qubit_count - 1 - index)) & 1
            for index in range(qubit_count)
        ]
        local_input = 2 * bits[first_qubit] + bits[second_qubit]
        for local_output in range(4):
            amplitude = local[local_output, local_input]
            if amplitude == 0.0:
                continue
            output_bits = bits.copy()
            output_bits[first_qubit] = local_output // 2
            output_bits[second_qubit] = local_output % 2
            row = 0
            for bit in output_bits:
                row = 2 * row + bit
            embedded[row, column] += amplitude
    return embedded


def _trace_qubit(density: Array, qubit_count: int, qubit: int) -> Array:
    """Trace one qubit from a density matrix in big-endian ordering."""

    dimension = 2**qubit_count
    if density.shape != (dimension, dimension):
        raise ValueError("density shape does not match qubit_count")
    if not 0 <= qubit < qubit_count:
        raise ValueError("qubit is out of range")
    tensor = density.reshape((2,) * (2 * qubit_count))
    reduced = np.trace(tensor, axis1=qubit, axis2=qubit_count + qubit)
    new_dimension = 2 ** (qubit_count - 1)
    return reduced.reshape(new_dimension, new_dimension)


def two_step_initial_state(parameters: TwoStepCollisionParameters) -> Array:
    """Ground thermometer and two independent thermal memory qubits."""

    parameters.validate()
    ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    thermal = thermal_state(parameters.temperature, parameters.energy_gap)
    return np.kron(np.kron(ground, thermal), thermal)


@lru_cache(maxsize=1024)
def _cycle_operators(
    temperature: float,
    memory_angle: float,
    system_memory_angle: float,
    energy_gap: float,
    probe_angle: float,
    measurement_polar_angle: float,
    measurement_azimuth: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    system_memory = _embedded_partial_swap(
        system_memory_angle, 3, 0, 1
    )
    identity_memory = np.eye(4, dtype=complex)
    kraus_zero = np.kron(
        readout_kraus(
            probe_angle,
            0,
            measurement_polar_angle,
            measurement_azimuth,
        ),
        identity_memory,
    )
    kraus_one = np.kron(
        readout_kraus(
            probe_angle,
            1,
            measurement_polar_angle,
            measurement_azimuth,
        ),
        identity_memory,
    )
    memory_one_two = _embedded_partial_swap(memory_angle, 4, 1, 2)
    memory_two_fresh = _embedded_partial_swap(memory_angle, 4, 2, 3)
    fresh = thermal_state(temperature, energy_gap)
    return (
        system_memory,
        kraus_zero,
        kraus_one,
        memory_one_two,
        memory_two_fresh,
        fresh,
    )


def _instrument_branches(
    density: Array,
    parameters: TwoStepCollisionParameters,
    probe_angle: float,
    false_positive: float,
    false_negative: float,
    measurement_polar_angle: float,
    measurement_azimuth: float,
) -> tuple[Array, Array]:
    operators = _cycle_operators(
        parameters.temperature,
        parameters.memory_angle,
        parameters.system_memory_angle,
        parameters.energy_gap,
        float(probe_angle),
        float(measurement_polar_angle),
        float(measurement_azimuth),
    )
    (
        system_memory,
        kraus_zero,
        kraus_one,
        memory_one_two,
        memory_two_fresh,
        fresh,
    ) = operators
    evolved = system_memory @ density @ system_memory.conj().T
    ideal_zero = kraus_zero @ evolved @ kraus_zero.conj().T
    ideal_one = kraus_one @ evolved @ kraus_one.conj().T
    reported = (
        (1.0 - false_positive) * ideal_zero + false_negative * ideal_one,
        false_positive * ideal_zero + (1.0 - false_negative) * ideal_one,
    )
    branches = []
    for observed in reported:
        enlarged = np.kron(observed, fresh)
        transferred = memory_one_two @ enlarged @ memory_one_two.conj().T
        transferred = (
            memory_two_fresh @ transferred @ memory_two_fresh.conj().T
        )
        branches.append(_trace_qubit(transferred, qubit_count=4, qubit=1))
    return branches[0], branches[1]


def two_step_record_distribution(
    parameters: TwoStepCollisionParameters,
    probe_angles: float | Sequence[float],
    length: int,
    *,
    false_positive: float = 0.0,
    false_negative: float = 0.0,
    measurement_polar_angles: float | Sequence[float] = 0.0,
    measurement_azimuths: float | Sequence[float] = 0.0,
) -> Array:
    """Enumerate the exact binary record law of the two-step memory model."""

    parameters.validate()
    _validate_assignment_errors(false_positive, false_negative)
    probe_schedule = normalize_schedule(probe_angles, length)
    polar_schedule, azimuth_schedule = normalize_measurement_schedule(
        measurement_polar_angles,
        measurement_azimuths,
        length,
    )
    branches: list[Array] = [two_step_initial_state(parameters)]
    for probe, polar, azimuth in zip(
        probe_schedule,
        polar_schedule,
        azimuth_schedule,
        strict=True,
    ):
        next_branches: list[Array] = []
        for branch in branches:
            next_branches.extend(
                _instrument_branches(
                    branch,
                    parameters,
                    float(probe),
                    false_positive,
                    false_negative,
                    float(polar),
                    float(azimuth),
                )
            )
        branches = next_branches
    probability = np.asarray([np.trace(branch).real for branch in branches])
    probability[np.abs(probability) < 1e-15] = 0.0
    if np.any(probability < -1e-11):
        raise FloatingPointError("negative record probability encountered")
    return np.maximum(probability, 0.0)


def two_step_assignment_fisher_matrix(
    parameters: TwoStepCollisionParameters,
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
    """Four-parameter Fisher matrix ordered as ``(T,mu,alpha,beta)``."""

    def distribution(
        candidate: TwoStepCollisionParameters,
        alpha: float,
        beta: float,
    ) -> Array:
        return two_step_record_distribution(
            candidate,
            probe_angles,
            length,
            false_positive=alpha,
            false_negative=beta,
            measurement_polar_angles=measurement_polar_angles,
            measurement_azimuths=measurement_azimuths,
        )

    base = distribution(parameters, false_positive, false_negative)
    derivatives = []
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
        derivatives.append(derivative)

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
            minus = base if lower < 0.0 else distribution(parameters, lower, beta)
        else:
            plus = distribution(parameters, alpha, upper)
            minus = base if lower < 0.0 else distribution(parameters, alpha, lower)
        denominator = error_step if lower < 0.0 else 2.0 * error_step
        derivatives.append((plus - minus) / denominator)

    jacobian = np.asarray(derivatives)
    mask = base > probability_floor
    selected = jacobian[:, mask]
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

