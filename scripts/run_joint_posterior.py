"""Compute long-record joint posteriors for temperature and memory strength."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.classical_full_swap import (
    pseudo_true_markov_temperature,
    sample_classical_record,
)
from memory_thermometry.model import CollisionParameters


def batched_instruments(
    temperatures: np.ndarray,
    memories: np.ndarray,
    coupling_angle: float,
    energy_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature_mesh, memory_mesh = np.meshgrid(
        temperatures, memories, indexing="ij"
    )
    excitation = 1.0 / (1.0 + np.exp(energy_gap / temperature_mesh))
    sin2_memory = np.sin(memory_mesh) ** 2
    cos2_memory = 1.0 - sin2_memory
    sin2_coupling = np.sin(coupling_angle) ** 2
    cos2_coupling = 1.0 - sin2_coupling
    a = excitation * cos2_memory
    b = a + sin2_memory

    shape = temperature_mesh.shape + (2, 2)
    matrix_zero = np.zeros(shape)
    matrix_one = np.zeros(shape)
    matrix_zero[..., 0, 0] = 1.0 - a
    matrix_zero[..., 0, 1] = cos2_coupling * (1.0 - b)
    matrix_zero[..., 1, 0] = a
    matrix_zero[..., 1, 1] = cos2_coupling * b
    matrix_one[..., 0, 1] = sin2_coupling * (1.0 - a)
    matrix_one[..., 1, 1] = sin2_coupling * a
    initial = np.stack([1.0 - excitation, excitation], axis=-1)
    return matrix_zero, matrix_one, initial


def posterior_summary(
    posterior: np.ndarray,
    temperatures: np.ndarray,
    memories: np.ndarray,
) -> dict[str, object]:
    temperature_mesh, memory_mesh = np.meshgrid(
        temperatures, memories, indexing="ij"
    )
    mean_temperature = float(np.sum(posterior * temperature_mesh))
    mean_memory = float(np.sum(posterior * memory_mesh))
    delta_temperature = temperature_mesh - mean_temperature
    delta_memory = memory_mesh - mean_memory
    variance_temperature = float(np.sum(posterior * delta_temperature**2))
    variance_memory = float(np.sum(posterior * delta_memory**2))
    covariance = float(np.sum(posterior * delta_temperature * delta_memory))
    correlation = covariance / np.sqrt(variance_temperature * variance_memory)
    map_index = np.unravel_index(np.argmax(posterior), posterior.shape)

    marginal_temperature = posterior.sum(axis=1)
    marginal_memory = posterior.sum(axis=0)

    def interval(grid: np.ndarray, marginal: np.ndarray) -> list[float]:
        cumulative = np.cumsum(marginal)
        return [
            float(grid[np.searchsorted(cumulative, 0.025)]),
            float(grid[np.searchsorted(cumulative, 0.975)]),
        ]

    return {
        "mean_temperature": mean_temperature,
        "mean_memory": mean_memory,
        "map_temperature": float(temperatures[map_index[0]]),
        "map_memory": float(memories[map_index[1]]),
        "temperature_sd": float(np.sqrt(variance_temperature)),
        "memory_sd": float(np.sqrt(variance_memory)),
        "posterior_correlation": float(correlation),
        "temperature_95_interval": interval(temperatures, marginal_temperature),
        "memory_95_interval": interval(memories, marginal_memory),
    }


def main() -> None:
    output = ROOT / "results" / "joint_posterior"
    output.mkdir(parents=True, exist_ok=True)

    true_parameters = CollisionParameters(
        temperature=0.90,
        memory_angle=0.50,
        system_memory_angle=0.55,
    )
    lengths = (100, 500, 2000, 10000)
    rng = np.random.default_rng(20260722)
    record = sample_classical_record(true_parameters, max(lengths), rng)

    temperatures = np.linspace(0.35, 3.00, 150)
    memories = np.linspace(0.0, 1.40, 141)
    matrix_zero, matrix_one, population = batched_instruments(
        temperatures,
        memories,
        true_parameters.system_memory_angle,
        true_parameters.energy_gap,
    )
    log_likelihood = np.zeros(population.shape[:-1])
    snapshots: dict[int, np.ndarray] = {}

    for index, outcome in enumerate(record, start=1):
        matrix = matrix_one if outcome else matrix_zero
        population = np.einsum("...ij,...j->...i", matrix, population)
        probability = population.sum(axis=-1)
        log_likelihood += np.log(np.maximum(probability, 1e-300))
        population /= probability[..., None]
        if index in lengths:
            shifted = log_likelihood - np.max(log_likelihood)
            posterior = np.exp(shifted)
            posterior /= posterior.sum()
            snapshots[index] = posterior.copy()

    summaries: dict[str, object] = {}
    for length, posterior in snapshots.items():
        summary = posterior_summary(posterior, temperatures, memories)
        summaries[str(length)] = summary
        fig, ax = plt.subplots(figsize=(6.2, 4.7), constrained_layout=True)
        image = ax.imshow(
            posterior,
            origin="lower",
            aspect="auto",
            extent=[
                memories.min(),
                memories.max(),
                temperatures.min(),
                temperatures.max(),
            ],
        )
        ax.plot(
            true_parameters.memory_angle,
            true_parameters.temperature,
            marker="x",
            markersize=8,
            markeredgewidth=2,
            color="white",
            label="true value",
        )
        ax.set_xlabel(r"memory angle $\mu$")
        ax.set_ylabel(r"temperature $T/\omega$")
        ax.set_title(f"Joint posterior from N={length} outcomes")
        ax.legend(loc="upper left")
        fig.colorbar(image, ax=ax, label="posterior mass per grid cell")
        fig.savefig(output / f"posterior_N{length}.png", dpi=180)
        plt.close(fig)

    analytic_markov_fit = pseudo_true_markov_temperature(true_parameters)
    output_summary = {
        "seed": 20260722,
        "true_parameters": {
            "temperature": true_parameters.temperature,
            "memory_angle": true_parameters.memory_angle,
            "system_memory_angle": true_parameters.system_memory_angle,
        },
        "record_excitation_fraction": float(record.mean()),
        "analytic_long_markov_fit_temperature": analytic_markov_fit,
        "snapshots": summaries,
    }
    np.savez(
        output / "posterior_data.npz",
        record=record,
        temperatures=temperatures,
        memories=memories,
        **{f"posterior_N{length}": posterior for length, posterior in snapshots.items()},
    )
    (output / "summary.json").write_text(
        json.dumps(output_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
