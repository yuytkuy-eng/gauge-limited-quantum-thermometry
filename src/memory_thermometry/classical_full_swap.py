"""Exact classical reduction for full-swap energy readout.

When the readout probe angle is pi/2, the thermometer is reset to its ground
state after every measurement. Starting from diagonal thermal states, the only
persistent degree of freedom is therefore the population of the memory qubit.
The complete quantum instrument reduces exactly to two nonnegative 2x2
matrices acting on an unnormalized memory-population vector.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .model import CollisionParameters


Array = np.ndarray


def _validate_assignment_errors(
    false_positive: float,
    false_negative: float,
    *,
    prefix: str = "",
) -> None:
    """Validate a binary assignment channel with positive determinant."""

    label = f"{prefix}_" if prefix else ""
    if not 0.0 <= false_positive < 1.0:
        raise ValueError(f"{label}false_positive must lie in [0, 1)")
    if not 0.0 <= false_negative < 1.0:
        raise ValueError(f"{label}false_negative must lie in [0, 1)")
    if false_positive + false_negative >= 1.0:
        raise ValueError(
            f"{label}false_positive + {label}false_negative must be less than 1"
        )


def _resolve_assignment_errors(
    readout_error: float,
    false_positive: float | None,
    false_negative: float | None,
) -> tuple[float, float]:
    """Resolve the legacy symmetric error or an asymmetric error pair."""

    if false_positive is None and false_negative is None:
        if not 0.0 <= readout_error < 0.5:
            raise ValueError("readout_error must lie in [0, 1/2)")
        return float(readout_error), float(readout_error)
    if false_positive is None or false_negative is None:
        raise ValueError(
            "false_positive and false_negative must be supplied together"
        )
    if readout_error != 0.0:
        raise ValueError(
            "readout_error cannot be combined with asymmetric assignment errors"
        )
    _validate_assignment_errors(false_positive, false_negative)
    return float(false_positive), float(false_negative)


def thermal_excitation_probability(
    temperature: float, energy_gap: float = 1.0
) -> float:
    """Excited-state Gibbs probability for a qubit."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if energy_gap <= 0.0:
        raise ValueError("energy_gap must be positive")
    return float(1.0 / (1.0 + np.exp(energy_gap / temperature)))


def temperature_from_excitation_probability(
    excitation: float, energy_gap: float = 1.0
) -> float:
    """Invert a qubit Gibbs population, including its zero-temperature limit."""

    if energy_gap <= 0.0:
        raise ValueError("energy_gap must be positive")
    if excitation < 0.0 or excitation >= 0.5:
        raise ValueError("excitation must lie in [0, 1/2)")
    if excitation == 0.0:
        return 0.0
    return float(energy_gap / np.log((1.0 - excitation) / excitation))


def full_swap_instrument_matrices(
    parameters: CollisionParameters,
    readout_error: float = 0.0,
    *,
    false_positive: float | None = None,
    false_negative: float | None = None,
) -> tuple[Array, Array]:
    """Return population instruments `(M_0, M_1)` for the two outcomes.

    Population vectors are columns ordered as `(ground, excited)`. Matrix
    columns correspond to the old-memory state and rows to the new-memory state.
    """

    parameters.validate()
    alpha, beta = _resolve_assignment_errors(
        readout_error, false_positive, false_negative
    )
    excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    sin2_memory = float(np.sin(parameters.memory_angle) ** 2)
    cos2_memory = 1.0 - sin2_memory
    sin2_coupling = float(np.sin(parameters.system_memory_angle) ** 2)
    cos2_coupling = 1.0 - sin2_coupling

    # New-memory excitation after the bath-memory collision, conditioned on
    # the old memory being respectively ground or excited.
    new_excitation_from_ground = excitation * cos2_memory
    new_excitation_from_excited = (
        new_excitation_from_ground + sin2_memory
    )

    matrix_zero = np.array(
        [
            [
                1.0 - new_excitation_from_ground,
                cos2_coupling * (1.0 - new_excitation_from_excited),
            ],
            [
                new_excitation_from_ground,
                cos2_coupling * new_excitation_from_excited,
            ],
        ],
        dtype=float,
    )
    matrix_one = np.array(
        [
            [0.0, sin2_coupling * (1.0 - new_excitation_from_ground)],
            [0.0, sin2_coupling * new_excitation_from_ground],
        ],
        dtype=float,
    )
    if alpha == 0.0 and beta == 0.0:
        return matrix_zero, matrix_one
    # Assignment matrix A_{observed,true} = [[1-alpha, beta],
    #                                         [alpha, 1-beta]].
    observed_zero = (1.0 - alpha) * matrix_zero + beta * matrix_one
    observed_one = alpha * matrix_zero + (1.0 - beta) * matrix_one
    return observed_zero, observed_one


def initial_population(parameters: CollisionParameters) -> Array:
    """Initial thermal-memory population vector."""

    excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    return np.array([1.0 - excitation, excitation], dtype=float)


def classical_record_distribution(
    parameters: CollisionParameters,
    length: int,
    readout_error: float = 0.0,
    *,
    false_positive: float | None = None,
    false_negative: float | None = None,
) -> Array:
    """Enumerate the full-swap record distribution using 2x2 instruments."""

    if length < 1:
        raise ValueError("length must be at least 1")
    matrices = full_swap_instrument_matrices(
        parameters,
        readout_error,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    branches = [initial_population(parameters)]
    for _ in range(length):
        next_branches: list[Array] = []
        for branch in branches:
            next_branches.append(matrices[0] @ branch)
            next_branches.append(matrices[1] @ branch)
        branches = next_branches
    return np.asarray([branch.sum() for branch in branches])


def classical_record_log_likelihood(
    parameters: CollisionParameters,
    record: Iterable[int],
    readout_error: float = 0.0,
    *,
    false_positive: float | None = None,
    false_negative: float | None = None,
) -> float:
    """Stable log likelihood for an arbitrarily long full-swap record."""

    matrices = full_swap_instrument_matrices(
        parameters,
        readout_error,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    population = initial_population(parameters)
    log_likelihood = 0.0
    for outcome in record:
        if outcome not in (0, 1):
            raise ValueError("record outcomes must be 0 or 1")
        population = matrices[int(outcome)] @ population
        probability = float(population.sum())
        if probability <= 0.0:
            return float("-inf")
        log_likelihood += float(np.log(probability))
        population /= probability
    return log_likelihood


def sample_classical_record(
    parameters: CollisionParameters,
    length: int,
    rng: np.random.Generator | None = None,
    readout_error: float = 0.0,
    *,
    false_positive: float | None = None,
    false_negative: float | None = None,
) -> Array:
    """Sample a full-swap binary record from the normalized population filter."""

    if length < 1:
        raise ValueError("length must be at least 1")
    if rng is None:
        rng = np.random.default_rng()
    matrices = full_swap_instrument_matrices(
        parameters,
        readout_error,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    population = initial_population(parameters)
    record = np.zeros(length, dtype=np.int8)
    for index in range(length):
        branches = [matrix @ population for matrix in matrices]
        probability_one = float(branches[1].sum())
        outcome = int(rng.random() < probability_one)
        probability = float(branches[outcome].sum())
        population = branches[outcome] / probability
        record[index] = outcome
    return record


def stationary_memory_excitation(parameters: CollisionParameters) -> float:
    """Unconditional stationary excitation before the system-memory collision."""

    parameters.validate()
    excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    sin2_memory = float(np.sin(parameters.memory_angle) ** 2)
    cos2_memory = 1.0 - sin2_memory
    cos2_coupling = float(np.cos(parameters.system_memory_angle) ** 2)
    denominator = 1.0 - cos2_coupling * sin2_memory
    return float(excitation * cos2_memory / denominator)


def finite_block_mean_memory_excitation(
    parameters: CollisionParameters, length: int
) -> float:
    """Average pre-collision memory excitation over a finite record block."""

    if length < 1:
        raise ValueError("length must be at least 1")
    excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    stationary = stationary_memory_excitation(parameters)
    ratio = float(
        np.cos(parameters.system_memory_angle) ** 2
        * np.sin(parameters.memory_angle) ** 2
    )
    if abs(1.0 - ratio) < 1e-14:
        transient_average = 1.0
    else:
        transient_average = (1.0 - ratio**length) / (length * (1.0 - ratio))
    return float(stationary + (excitation - stationary) * transient_average)


def pseudo_true_markov_temperature(
    parameters: CollisionParameters, length: int | None = None
) -> float:
    """Temperature selected by a misspecified Markov likelihood.

    With `length=None`, return the infinite-record limit. With a positive block
    length, return the exact pseudo-true value for repeated independent blocks
    of that length.
    """

    if length is None:
        effective_excitation = stationary_memory_excitation(parameters)
    else:
        effective_excitation = finite_block_mean_memory_excitation(
            parameters, length
        )
    return temperature_from_excitation_probability(
        effective_excitation, parameters.energy_gap
    )


def pseudo_true_temperature_with_readout_error(
    parameters: CollisionParameters,
    true_readout_error: float,
    assumed_readout_error: float,
    length: int | None = None,
) -> float:
    """Pseudo-true Markov temperature under calibrated or miscalibrated readout.

    The data are generated with environmental memory and `true_readout_error`.
    The fitted model assumes a Markov bath and `assumed_readout_error`. A return
    value of infinity means the optimum is at the infinite-temperature boundary.
    """

    return pseudo_true_temperature_with_assignment_error(
        parameters,
        true_false_positive=true_readout_error,
        true_false_negative=true_readout_error,
        assumed_false_positive=assumed_readout_error,
        assumed_false_negative=assumed_readout_error,
        length=length,
    )


def pseudo_true_temperature_with_assignment_error(
    parameters: CollisionParameters,
    true_false_positive: float,
    true_false_negative: float,
    assumed_false_positive: float,
    assumed_false_negative: float,
    length: int | None = None,
) -> float:
    """Pseudo-true Markov temperature under a general assignment channel.

    ``false_positive`` is P(observed 1 | true 0), while ``false_negative`` is
    P(observed 0 | true 1). The true and fitted detectors can be independently
    specified. A return value of infinity is the infinite-temperature boundary.
    """

    _validate_assignment_errors(
        true_false_positive, true_false_negative, prefix="true"
    )
    _validate_assignment_errors(
        assumed_false_positive, assumed_false_negative, prefix="assumed"
    )

    if length is None:
        memory_excitation = stationary_memory_excitation(parameters)
    else:
        memory_excitation = finite_block_mean_memory_excitation(parameters, length)

    click_scale = float(np.sin(parameters.system_memory_angle) ** 2)
    if click_scale <= 0.0:
        raise ValueError("system_memory_angle must give a nonzero click probability")
    true_click = true_false_positive + (
        1.0 - true_false_positive - true_false_negative
    ) * click_scale * memory_excitation
    denominator = (
        1.0 - assumed_false_positive - assumed_false_negative
    ) * click_scale
    fitted_excitation = (
        true_click - assumed_false_positive
    ) / denominator
    if fitted_excitation <= 0.0:
        return 0.0
    if fitted_excitation >= 0.5:
        return float("inf")
    return temperature_from_excitation_probability(
        float(fitted_excitation), parameters.energy_gap
    )


def critical_ignored_readout_error(
    parameters: CollisionParameters, length: int | None = None
) -> float:
    """Unmodeled symmetric bit-flip rate at which temperature bias changes sign."""

    thermal_excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    if length is None:
        memory_excitation = stationary_memory_excitation(parameters)
    else:
        memory_excitation = finite_block_mean_memory_excitation(parameters, length)
    click_scale = float(np.sin(parameters.system_memory_angle) ** 2)
    numerator = click_scale * (thermal_excitation - memory_excitation)
    denominator = 1.0 - 2.0 * click_scale * memory_excitation
    return float(numerator / denominator)


def critical_ignored_false_positive_rate(
    parameters: CollisionParameters,
    false_negative: float,
    length: int | None = None,
) -> float:
    """False-positive rate cancelling memory bias for an ideal-detector fit.

    The data have the supplied false-negative rate, whereas the fitted Markov
    model ignores both assignment errors. Values outside the admissible region
    ``alpha >= 0`` and ``alpha + beta < 1`` mean cancellation is unreachable
    for a detector with positive assignment contrast.
    """

    if not 0.0 <= false_negative < 1.0:
        raise ValueError("false_negative must lie in [0, 1)")
    thermal_excitation = thermal_excitation_probability(
        parameters.temperature, parameters.energy_gap
    )
    if length is None:
        memory_excitation = stationary_memory_excitation(parameters)
    else:
        memory_excitation = finite_block_mean_memory_excitation(parameters, length)
    click_scale = float(np.sin(parameters.system_memory_angle) ** 2)
    numerator = click_scale * (
        thermal_excitation - memory_excitation
        + false_negative * memory_excitation
    )
    denominator = 1.0 - click_scale * memory_excitation
    return float(numerator / denominator)
