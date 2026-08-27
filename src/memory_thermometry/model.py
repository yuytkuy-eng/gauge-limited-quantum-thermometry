"""Minimal collision model for thermometry with one-step bath memory.

The persistent Hilbert space is S x M, with both subsystems qubits. The old
memory transfers information to a fresh Gibbs qubit after every observed probe
collision. Binary record distributions are calculated exactly by branching the
unnormalized conditional density matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np


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


def _kron(left: Array, right: Array) -> Array:
    """Kronecker product for matrices, kept local for hot-loop stability."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("_kron expects two matrices")
    return np.einsum("ij,kl->ikjl", left, right).reshape(
        left.shape[0] * right.shape[0],
        left.shape[1] * right.shape[1],
    )


@dataclass(frozen=True)
class CollisionParameters:
    """Physical parameters in units where k_B = hbar = 1."""

    temperature: float
    memory_angle: float
    system_memory_angle: float = 0.55
    energy_gap: float = 1.0

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.energy_gap <= 0:
            raise ValueError("energy_gap must be positive")
        for name, angle in (
            ("memory_angle", self.memory_angle),
            ("system_memory_angle", self.system_memory_angle),
        ):
            if not 0.0 <= angle <= 0.5 * np.pi:
                raise ValueError(f"{name} must lie in [0, pi/2]")


def thermal_state(temperature: float, energy_gap: float = 1.0) -> Array:
    """Return a qubit Gibbs state with |0> as the ground state."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    excited = 1.0 / (1.0 + np.exp(energy_gap / temperature))
    return np.diag([1.0 - excited, excited]).astype(complex)


def partial_swap_xy(angle: float) -> Array:
    """Excitation-preserving partial swap generated in the one-excitation block."""

    c = np.cos(angle)
    s = np.sin(angle)
    unitary = np.eye(4, dtype=complex)
    unitary[1, 1] = c
    unitary[2, 2] = c
    unitary[1, 2] = -1j * s
    unitary[2, 1] = -1j * s
    return unitary


def readout_kraus(
    probe_angle: float,
    outcome: int,
    measurement_polar_angle: float = 0.0,
    measurement_azimuth: float = 0.0,
) -> Array:
    """Kraus operator induced by measuring the ground-state readout probe.

    ``measurement_polar_angle=0`` recovers the energy-basis detector used in
    the original model.  Nonzero polar angles describe a known single-qubit
    rotation immediately before that same binary detector.  The ideal probe
    measurement kets are

    ``|m0> = cos(phi/2)|0> + exp(i chi) sin(phi/2)|1>`` and
    ``|m1> = -exp(-i chi) sin(phi/2)|0> + cos(phi/2)|1>``.
    """

    if not 0.0 <= probe_angle <= 0.5 * np.pi:
        raise ValueError("probe_angle must lie in [0, pi/2]")
    if not 0.0 <= measurement_polar_angle <= np.pi:
        raise ValueError("measurement_polar_angle must lie in [0, pi]")
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")

    half = 0.5 * measurement_polar_angle
    cosine = np.cos(half)
    sine = np.sin(half)
    if outcome == 0:
        measurement_ket = np.array(
            [cosine, np.exp(1j * measurement_azimuth) * sine], dtype=complex
        )
    else:
        measurement_ket = np.array(
            [-np.exp(-1j * measurement_azimuth) * sine, cosine], dtype=complex
        )
    amplitude_zero, amplitude_one = measurement_ket.conj()
    return np.array(
        [
            [
                amplitude_zero,
                -1j * np.sin(probe_angle) * amplitude_one,
            ],
            [0.0, np.cos(probe_angle) * amplitude_zero],
        ],
        dtype=complex,
    )


def _trace_old_memory(rho_sme: Array) -> Array:
    """Trace M from a density matrix ordered as S x M x E, keeping S x E."""

    tensor = rho_sme.reshape(2, 2, 2, 2, 2, 2)
    reduced = np.trace(tensor, axis1=1, axis2=4)
    return reduced.reshape(4, 4)


def initial_state(parameters: CollisionParameters) -> Array:
    """Ground-state thermometer with a fresh thermal memory qubit."""

    parameters.validate()
    ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    return _kron(
        ground, thermal_state(parameters.temperature, parameters.energy_gap)
    )


@lru_cache(maxsize=512)
def _cycle_operators(
    temperature: float,
    memory_angle: float,
    system_memory_angle: float,
    energy_gap: float,
    probe_angle: float,
    measurement_polar_angle: float,
    measurement_azimuth: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Cache operators reused by every branch at one parameter point."""

    u_sm = partial_swap_xy(system_memory_angle)
    identity = np.eye(2, dtype=complex)
    k_zero = _kron(
        readout_kraus(
            probe_angle,
            0,
            measurement_polar_angle,
            measurement_azimuth,
        ),
        identity,
    )
    k_one = _kron(
        readout_kraus(
            probe_angle,
            1,
            measurement_polar_angle,
            measurement_azimuth,
        ),
        identity,
    )
    fresh = thermal_state(temperature, energy_gap)
    u_sme = _kron(identity, partial_swap_xy(memory_angle))
    return u_sm, k_zero, k_one, fresh, u_sme


def _instrument_branches(
    rho_sm: Array,
    parameters: CollisionParameters,
    probe_angle: float,
    false_positive: float,
    false_negative: float,
    measurement_polar_angle: float = 0.0,
    measurement_azimuth: float = 0.0,
) -> tuple[Array, Array]:
    """Return both reported-outcome branches while sharing linear algebra."""

    u_sm, k_zero, k_one, fresh, u_sme = _cycle_operators(
        parameters.temperature,
        parameters.memory_angle,
        parameters.system_memory_angle,
        parameters.energy_gap,
        float(probe_angle),
        float(measurement_polar_angle),
        float(measurement_azimuth),
    )
    evolved = u_sm @ rho_sm @ u_sm.conj().T
    ideal_zero = k_zero @ evolved @ k_zero.conj().T
    ideal_one = k_one @ evolved @ k_one.conj().T
    reported = (
        (1.0 - false_positive) * ideal_zero + false_negative * ideal_one,
        false_positive * ideal_zero + (1.0 - false_negative) * ideal_one,
    )
    transferred_branches = []
    for observed in reported:
        rho_sme = _kron(observed, fresh)
        transferred = u_sme @ rho_sme @ u_sme.conj().T
        transferred_branches.append(_trace_old_memory(transferred))
    return transferred_branches[0], transferred_branches[1]


@lru_cache(maxsize=4096)
def _instrument_superoperators(
    temperature: float,
    memory_angle: float,
    system_memory_angle: float,
    energy_gap: float,
    probe_angle: float,
    false_positive: float,
    false_negative: float,
    measurement_polar_angle: float,
    measurement_azimuth: float,
) -> tuple[Array, Array]:
    """Vectorized conditional maps for one controlled cycle."""

    u_sm, k_zero, k_one, fresh, u_sme = _cycle_operators(
        temperature,
        memory_angle,
        system_memory_angle,
        energy_gap,
        probe_angle,
        measurement_polar_angle,
        measurement_azimuth,
    )
    basis = np.eye(16, dtype=complex).reshape(16, 4, 4, order="F")
    enlarged = np.einsum("nab,cd->nacbd", basis, fresh).reshape(16, 8, 8)
    transferred = np.einsum(
        "ab,nbc,cd->nad", u_sme, enlarged, u_sme.conj().T
    )
    tensor = transferred.reshape(16, 2, 2, 2, 2, 2, 2)
    reduced = np.trace(tensor, axis1=2, axis2=5).reshape(16, 4, 4)
    transfer = np.stack(
        [matrix.reshape(-1, order="F") for matrix in reduced], axis=1
    )

    evolved_zero = k_zero @ u_sm
    evolved_one = k_one @ u_sm
    ideal_zero = np.kron(evolved_zero.conj(), evolved_zero)
    ideal_one = np.kron(evolved_one.conj(), evolved_one)
    reported_zero = (
        (1.0 - false_positive) * ideal_zero
        + false_negative * ideal_one
    )
    reported_one = (
        false_positive * ideal_zero
        + (1.0 - false_negative) * ideal_one
    )
    return transfer @ reported_zero, transfer @ reported_one


def instrument_step(
    rho_sm: Array,
    parameters: CollisionParameters,
    probe_angle: float,
    outcome: int,
    *,
    false_positive: float = 0.0,
    false_negative: float = 0.0,
    measurement_polar_angle: float = 0.0,
    measurement_azimuth: float = 0.0,
) -> Array:
    """Apply one conditional cycle to an unnormalized S x M state.

    The returned S x M state is unnormalized. Its trace is the joint
    probability weight accumulated along the corresponding measurement branch.
    """

    parameters.validate()
    _validate_assignment_errors(false_positive, false_negative)
    if rho_sm.shape != (4, 4):
        raise ValueError("rho_sm must have shape (4, 4)")
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")

    return _instrument_branches(
        rho_sm,
        parameters,
        probe_angle,
        false_positive,
        false_negative,
        measurement_polar_angle,
        measurement_azimuth,
    )[outcome]


def normalize_schedule(probe_angles: float | Sequence[float], length: int) -> Array:
    """Expand a scalar or periodic probe-angle schedule to the record length."""

    if length < 1:
        raise ValueError("length must be at least 1")
    if np.isscalar(probe_angles):
        schedule = np.full(length, float(probe_angles), dtype=float)
    else:
        base = np.asarray(list(probe_angles), dtype=float)
        if base.ndim != 1 or base.size == 0:
            raise ValueError("probe_angles must be a nonempty one-dimensional sequence")
        schedule = np.resize(base, length)
    if np.any(schedule < 0.0) or np.any(schedule > 0.5 * np.pi):
        raise ValueError("all probe angles must lie in [0, pi/2]")
    return schedule


def _normalize_measurement_schedule(
    values: float | Sequence[float],
    length: int,
    name: str,
) -> Array:
    """Expand a scalar or periodic measurement-control schedule."""

    if np.isscalar(values):
        return np.full(length, float(values), dtype=float)
    base = np.asarray(list(values), dtype=float)
    if base.ndim != 1 or base.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    return np.resize(base, length)


def normalize_measurement_schedule(
    measurement_polar_angles: float | Sequence[float],
    measurement_azimuths: float | Sequence[float],
    length: int,
) -> tuple[Array, Array]:
    """Expand the known measurement-basis controls to the record length."""

    if length < 1:
        raise ValueError("length must be at least 1")
    polar = _normalize_measurement_schedule(
        measurement_polar_angles,
        length,
        "measurement_polar_angles",
    )
    azimuth = _normalize_measurement_schedule(
        measurement_azimuths,
        length,
        "measurement_azimuths",
    )
    if np.any(polar < 0.0) or np.any(polar > np.pi):
        raise ValueError("all measurement polar angles must lie in [0, pi]")
    return polar, azimuth


def record_distribution(
    parameters: CollisionParameters,
    probe_angles: float | Sequence[float],
    length: int,
    *,
    false_positive: float = 0.0,
    false_negative: float = 0.0,
    measurement_polar_angles: float | Sequence[float] = 0.0,
    measurement_azimuths: float | Sequence[float] = 0.0,
) -> Array:
    """Enumerate the exact distribution of all binary records of given length.

    Branch ordering is lexicographic in the generated bits: after one step the
    entries correspond to records `0, 1`; after two steps to `00, 01, 10, 11`.
    """

    parameters.validate()
    _validate_assignment_errors(false_positive, false_negative)
    schedule = normalize_schedule(probe_angles, length)
    polar_schedule, azimuth_schedule = normalize_measurement_schedule(
        measurement_polar_angles,
        measurement_azimuths,
        length,
    )
    branches: list[Array] = [initial_state(parameters)]
    for angle, polar, azimuth in zip(
        schedule,
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
                    float(angle),
                    false_positive,
                    false_negative,
                    float(polar),
                    float(azimuth),
                )
            )
        branches = next_branches

    probabilities = np.asarray([np.trace(rho).real for rho in branches])
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    if np.any(probabilities < -1e-11):
        raise FloatingPointError("negative record probability encountered")
    probabilities = np.maximum(probabilities, 0.0)
    return probabilities


def record_distribution_superoperator(
    parameters: CollisionParameters,
    probe_angles: float | Sequence[float],
    length: int,
    *,
    false_positive: float = 0.0,
    false_negative: float = 0.0,
    measurement_polar_angles: float | Sequence[float] = 0.0,
    measurement_azimuths: float | Sequence[float] = 0.0,
) -> Array:
    """Enumerate record probabilities using cached 16x16 superoperators.

    This is algebraically equivalent to :func:`record_distribution` but is
    substantially faster when the same control schedule is evaluated at many
    parameter points, as in four-dimensional posterior sampling.
    """

    parameters.validate()
    _validate_assignment_errors(false_positive, false_negative)
    schedule = normalize_schedule(probe_angles, length)
    polar_schedule, azimuth_schedule = normalize_measurement_schedule(
        measurement_polar_angles,
        measurement_azimuths,
        length,
    )
    vector = initial_state(parameters).reshape(-1, order="F")
    branches = vector[None, :]
    for angle, polar, azimuth in zip(
        schedule, polar_schedule, azimuth_schedule, strict=True
    ):
        superoperators = _instrument_superoperators(
            parameters.temperature,
            parameters.memory_angle,
            parameters.system_memory_angle,
            parameters.energy_gap,
            float(angle),
            false_positive,
            false_negative,
            float(polar),
            float(azimuth),
        )
        branches = np.concatenate(
            [branches @ operator.T for operator in superoperators], axis=0
        ).reshape(2, -1, 16).transpose(1, 0, 2).reshape(-1, 16)
    trace_vector = np.eye(4, dtype=complex).reshape(-1, order="F").conj()
    probabilities = np.real(branches @ trace_vector)
    probabilities[np.abs(probabilities) < 1e-15] = 0.0
    if np.any(probabilities < -1e-11):
        raise FloatingPointError("negative record probability encountered")
    return np.maximum(probabilities, 0.0)


def record_bits(index: int, length: int) -> tuple[int, ...]:
    """Convert a distribution index to its binary measurement record."""

    if not 0 <= index < 2**length:
        raise ValueError("index outside the record range")
    return tuple(int(bit) for bit in f"{index:0{length}b}")


def record_log_likelihood(
    parameters: CollisionParameters,
    probe_angles: float | Sequence[float],
    record: Iterable[int],
    *,
    false_positive: float = 0.0,
    false_negative: float = 0.0,
    measurement_polar_angles: float | Sequence[float] = 0.0,
    measurement_azimuths: float | Sequence[float] = 0.0,
) -> float:
    """Stable log likelihood for a general, possibly long, measurement record."""

    outcomes = tuple(int(outcome) for outcome in record)
    if not outcomes:
        raise ValueError("record must be nonempty")
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("record outcomes must be 0 or 1")
    parameters.validate()
    _validate_assignment_errors(false_positive, false_negative)
    schedule = normalize_schedule(probe_angles, len(outcomes))
    polar_schedule, azimuth_schedule = normalize_measurement_schedule(
        measurement_polar_angles,
        measurement_azimuths,
        len(outcomes),
    )
    state = initial_state(parameters)
    log_likelihood = 0.0
    for angle, polar, azimuth, outcome in zip(
        schedule,
        polar_schedule,
        azimuth_schedule,
        outcomes,
        strict=True,
    ):
        state = _instrument_branches(
            state,
            parameters,
            float(angle),
            false_positive,
            false_negative,
            float(polar),
            float(azimuth),
        )[outcome]
        probability = float(np.trace(state).real)
        if probability <= 0.0:
            return float("-inf")
        log_likelihood += float(np.log(probability))
        state /= probability
    return log_likelihood


def sample_record(
    parameters: CollisionParameters,
    probe_angles: float | Sequence[float],
    length: int,
    rng: np.random.Generator | None = None,
    *,
    false_positive: float = 0.0,
    false_negative: float = 0.0,
    measurement_polar_angles: float | Sequence[float] = 0.0,
    measurement_azimuths: float | Sequence[float] = 0.0,
) -> Array:
    """Sample a binary trajectory from the general quantum instrument."""

    parameters.validate()
    _validate_assignment_errors(false_positive, false_negative)
    schedule = normalize_schedule(probe_angles, length)
    polar_schedule, azimuth_schedule = normalize_measurement_schedule(
        measurement_polar_angles,
        measurement_azimuths,
        length,
    )
    if rng is None:
        rng = np.random.default_rng()
    state = initial_state(parameters)
    record = np.zeros(length, dtype=np.int8)
    for index, (angle, polar, azimuth) in enumerate(
        zip(schedule, polar_schedule, azimuth_schedule, strict=True)
    ):
        branches = _instrument_branches(
            state,
            parameters,
            float(angle),
            false_positive,
            false_negative,
            float(polar),
            float(azimuth),
        )
        probability_one = float(np.trace(branches[1]).real)
        outcome = int(rng.random() < probability_one)
        probability = float(np.trace(branches[outcome]).real)
        state = branches[outcome] / probability
        record[index] = outcome
    return record
