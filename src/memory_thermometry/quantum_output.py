"""Joint output-probe states and their quantum Fisher information.

At full probe swap the thermometer is reset after every cycle. The outgoing
probe is the pre-reset thermometer state up to a fixed local phase, so the
unmeasured output can be propagated while retaining only the accumulated
probes and the current bath-memory qubit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .assignment_fisher import effective_target_information
from .model import CollisionParameters, partial_swap_xy, thermal_state


Array = np.ndarray


@dataclass(frozen=True)
class QuantumOutputFisherResult:
    """QFI for parameters ``(temperature, memory_angle)``."""

    matrix: Array
    effective_temperature_information: float
    eigenvalues: Array
    state_dimension: int
    trace_error: float
    minimum_state_eigenvalue: float
    mean_sld_commutator: Array


def _apply_last_subsystem_unitary(
    density: Array,
    history_dimension: int,
    local_dimension: int,
    unitary: Array,
) -> Array:
    """Apply ``I_history tensor unitary`` without forming the large identity."""

    expected = history_dimension * local_dimension
    if density.shape != (expected, expected):
        raise ValueError("density shape does not match the subsystem dimensions")
    if unitary.shape != (local_dimension, local_dimension):
        raise ValueError("unitary shape does not match local_dimension")
    tensor = density.reshape(
        history_dimension,
        local_dimension,
        history_dimension,
        local_dimension,
    )
    transformed = np.einsum(
        "ab,ibjc,cd->iajd",
        unitary,
        tensor,
        unitary.conj().T,
        optimize=True,
    )
    return transformed.reshape(expected, expected)


def _append_ground_before_memory(density: Array, history_dimension: int) -> Array:
    """Map ``history x M`` to ``history x S x M`` with ground-state ``S``."""

    if density.shape != (2 * history_dimension, 2 * history_dimension):
        raise ValueError("density must describe history tensor memory")
    memory_tensor = density.reshape(history_dimension, 2, history_dimension, 2)
    ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    expanded = np.einsum(
        "imjn,st->ismjtn",
        memory_tensor,
        ground,
        optimize=True,
    )
    return expanded.reshape(4 * history_dimension, 4 * history_dimension)


def _transfer_memory(
    density: Array,
    probe_dimension: int,
    fresh_memory: Array,
    memory_unitary: Array,
) -> Array:
    """Interact old memory with a fresh qubit, trace old, and keep the fresh."""

    if density.shape != (2 * probe_dimension, 2 * probe_dimension):
        raise ValueError("density must describe probes tensor old memory")
    enlarged = np.kron(density, fresh_memory)
    transferred = _apply_last_subsystem_unitary(
        enlarged,
        history_dimension=probe_dimension,
        local_dimension=4,
        unitary=memory_unitary,
    )
    tensor = transferred.reshape(
        probe_dimension,
        2,
        2,
        probe_dimension,
        2,
        2,
    )
    reduced = np.trace(tensor, axis1=1, axis2=4)
    return reduced.reshape(2 * probe_dimension, 2 * probe_dimension)


def full_swap_probe_output_state(
    parameters: CollisionParameters,
    length: int,
) -> Array:
    """Return the joint state of all unmeasured full-swap output probes."""

    parameters.validate()
    if length < 1:
        raise ValueError("length must be at least 1")

    density = thermal_state(parameters.temperature, parameters.energy_gap)
    system_memory = partial_swap_xy(parameters.system_memory_angle)
    memory_transfer = partial_swap_xy(parameters.memory_angle)
    fresh = thermal_state(parameters.temperature, parameters.energy_gap)
    # A full XY swap transfers |1>_S to -i|1>_P. This fixed phase does not
    # change QFI, but including it returns the physical outgoing-probe state.
    output_phase = np.diag([1.0, -1j]).astype(complex)
    output_phase_memory = np.kron(output_phase, np.eye(2, dtype=complex))

    probe_dimension = 1
    for _ in range(length):
        with_system = _append_ground_before_memory(density, probe_dimension)
        after_collision = _apply_last_subsystem_unitary(
            with_system,
            history_dimension=probe_dimension,
            local_dimension=4,
            unitary=system_memory,
        )
        phased_output = _apply_last_subsystem_unitary(
            after_collision,
            history_dimension=probe_dimension,
            local_dimension=4,
            unitary=output_phase_memory,
        )
        probe_dimension *= 2
        density = _transfer_memory(
            phased_output,
            probe_dimension,
            fresh,
            memory_transfer,
        )

    tensor = density.reshape(probe_dimension, 2, probe_dimension, 2)
    output = np.trace(tensor, axis1=1, axis2=3)
    output = 0.5 * (output + output.conj().T)
    output /= np.trace(output).real
    return output


def _state_derivative(
    parameters: CollisionParameters,
    field: str,
    step: float,
    length: int,
) -> Array:
    value = float(getattr(parameters, field))
    lower = value - step
    upper = value + step
    lower_invalid = (field == "temperature" and lower <= 0.0) or (
        field == "memory_angle" and lower < 0.0
    )
    upper_invalid = field == "memory_angle" and upper > 0.5 * np.pi
    base = None
    if lower_invalid:
        base = full_swap_probe_output_state(parameters, length)
        plus = full_swap_probe_output_state(
            replace(parameters, **{field: upper}), length
        )
        return (plus - base) / step
    if upper_invalid:
        base = full_swap_probe_output_state(parameters, length)
        minus = full_swap_probe_output_state(
            replace(parameters, **{field: lower}), length
        )
        return (base - minus) / step
    plus = full_swap_probe_output_state(
        replace(parameters, **{field: upper}), length
    )
    minus = full_swap_probe_output_state(
        replace(parameters, **{field: lower}), length
    )
    return (plus - minus) / (2.0 * step)


def density_matrix_quantum_fisher(
    density: Array,
    derivatives: Array,
    eigenvalue_floor: float = 1e-12,
) -> Array:
    """Return the symmetric-logarithmic-derivative QFI matrix."""

    state = np.asarray(density, dtype=complex)
    derivative = np.asarray(derivatives, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("density must be square")
    if derivative.ndim != 3 or derivative.shape[1:] != state.shape:
        raise ValueError("derivatives must have shape (parameters, d, d)")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (state + state.conj().T))
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    weight = np.zeros_like(denominator)
    mask = denominator > eigenvalue_floor
    weight[mask] = 2.0 / denominator[mask]
    transformed = np.asarray(
        [eigenvectors.conj().T @ item @ eigenvectors for item in derivative]
    )
    count = derivative.shape[0]
    fisher = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(first, count):
            value = np.sum(
                weight
                * np.real(transformed[first] * transformed[second].conj())
            )
            fisher[first, second] = float(value)
            fisher[second, first] = float(value)
    return fisher


def symmetric_logarithmic_derivatives(
    density: Array,
    derivatives: Array,
    eigenvalue_floor: float = 1e-12,
) -> Array:
    """Construct symmetric logarithmic derivatives for a state family."""

    state = np.asarray(density, dtype=complex)
    derivative = np.asarray(derivatives, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("density must be square")
    if derivative.ndim != 3 or derivative.shape[1:] != state.shape:
        raise ValueError("derivatives must have shape (parameters, d, d)")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (state + state.conj().T))
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    mask = denominator > eigenvalue_floor
    logarithmic_derivatives = []
    for item in derivative:
        transformed = eigenvectors.conj().T @ item @ eigenvectors
        sld_eigenbasis = np.zeros_like(transformed)
        sld_eigenbasis[mask] = 2.0 * transformed[mask] / denominator[mask]
        sld = eigenvectors @ sld_eigenbasis @ eigenvectors.conj().T
        logarithmic_derivatives.append(0.5 * (sld + sld.conj().T))
    return np.asarray(logarithmic_derivatives)


def mean_sld_commutator_matrix(density: Array, slds: Array) -> Array:
    """Return ``Tr[rho [L_i,L_j]]/(2i)`` as a real antisymmetric matrix."""

    state = np.asarray(density, dtype=complex)
    logarithmic_derivatives = np.asarray(slds, dtype=complex)
    if logarithmic_derivatives.ndim != 3:
        raise ValueError("slds must have shape (parameters, d, d)")
    count = logarithmic_derivatives.shape[0]
    result = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(first + 1, count):
            commutator = (
                logarithmic_derivatives[first] @ logarithmic_derivatives[second]
                - logarithmic_derivatives[second] @ logarithmic_derivatives[first]
            )
            value = np.trace(state @ commutator) / (2j)
            result[first, second] = float(np.real_if_close(value))
            result[second, first] = -result[first, second]
    return result


def full_swap_probe_quantum_fisher(
    parameters: CollisionParameters,
    length: int,
    temperature_step: float = 1e-4,
    memory_step: float = 1e-4,
    eigenvalue_floor: float = 1e-12,
) -> QuantumOutputFisherResult:
    """Compute the two-parameter QFI of the joint output-probe state."""

    density = full_swap_probe_output_state(parameters, length)
    derivatives = np.asarray(
        [
            _state_derivative(parameters, "temperature", temperature_step, length),
            _state_derivative(parameters, "memory_angle", memory_step, length),
        ]
    )
    matrix = density_matrix_quantum_fisher(
        density, derivatives, eigenvalue_floor=eigenvalue_floor
    )
    slds = symmetric_logarithmic_derivatives(
        density, derivatives, eigenvalue_floor=eigenvalue_floor
    )
    commutator = mean_sld_commutator_matrix(density, slds)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return QuantumOutputFisherResult(
        matrix=matrix,
        effective_temperature_information=effective_target_information(matrix),
        eigenvalues=eigenvalues,
        state_dimension=density.shape[0],
        trace_error=float(abs(np.trace(density).real - 1.0)),
        minimum_state_eigenvalue=float(np.linalg.eigvalsh(density)[0]),
        mean_sld_commutator=commutator,
    )
