"""Risk utilities for Bayes and minimax c-optimal experimental design."""

from __future__ import annotations

from itertools import product

import numpy as np


Array = np.ndarray


def c_optimal_variance(
    component_information: Array,
    weights: Array,
    target_index: int = 0,
) -> Array:
    """Return ``c.T @ M(theta,w)^-1 @ c`` for every parameter point.

    ``component_information`` may have shape ``(settings,p,p)`` or
    ``(points,settings,p,p)``. The weights are resource fractions and must lie
    on the simplex. Singular mixtures have infinite risk.
    """

    matrices = np.asarray(component_information, dtype=float)
    allocation = np.asarray(weights, dtype=float)
    if matrices.ndim not in (3, 4) or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "component_information must have shape (settings,p,p) or "
            "(points,settings,p,p)"
        )
    setting_axis = 0 if matrices.ndim == 3 else 1
    if allocation.shape != (matrices.shape[setting_axis],):
        raise ValueError("one weight is required per design setting")
    if np.any(allocation < 0.0) or not np.isclose(allocation.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    parameter_count = matrices.shape[-1]
    if not 0 <= target_index < parameter_count:
        raise ValueError("target_index is out of range")
    if matrices.ndim == 3:
        information = np.einsum("k,kij->ij", allocation, matrices)
        try:
            variance = np.linalg.solve(
                information,
                np.eye(parameter_count)[:, target_index],
            )[target_index]
        except np.linalg.LinAlgError:
            return np.asarray(float("inf"))
        return np.asarray(float(variance))
    information = np.einsum("k,gkij->gij", allocation, matrices)
    result = np.empty(information.shape[0])
    target = np.eye(parameter_count)[:, target_index]
    for index, matrix in enumerate(information):
        try:
            result[index] = np.linalg.solve(matrix, target)[target_index]
        except np.linalg.LinAlgError:
            result[index] = float("inf")
    return result


def relative_c_efficiency(
    variance: Array, oracle_variance: Array
) -> Array:
    """Return pointwise oracle variance divided by candidate variance."""

    candidate = np.asarray(variance, dtype=float)
    oracle = np.asarray(oracle_variance, dtype=float)
    if candidate.shape != oracle.shape:
        raise ValueError("variance arrays must have the same shape")
    if np.any(candidate <= 0.0) or np.any(oracle <= 0.0):
        raise ValueError("variances must be positive")
    return oracle / candidate


def trapezoidal_axis_weights(axis: Array) -> Array:
    """Normalized one-dimensional trapezoidal quadrature weights."""

    values = np.asarray(axis, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("axis must contain at least two points")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("axis must be strictly increasing")
    differences = np.diff(values)
    weights = np.empty(values.size)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    if values.size > 2:
        weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return weights / weights.sum()


def tensor_product_weights(*axes: Array) -> Array:
    """Flatten normalized tensor-product trapezoidal weights lexicographically."""

    if not axes:
        raise ValueError("at least one axis is required")
    one_dimensional = [trapezoidal_axis_weights(axis) for axis in axes]
    result = np.asarray(
        [np.prod(values) for values in product(*one_dimensional)],
        dtype=float,
    )
    return result / result.sum()
