"""Coarse optimization of two-setting probe schedules at a reversal point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.fisher import classical_fisher_matrix
from memory_thermometry.model import CollisionParameters


def evaluate(
    parameters: CollisionParameters,
    schedule: float | tuple[float, ...],
    length: int,
) -> tuple[float, float, float]:
    result = classical_fisher_matrix(parameters, schedule, length)
    return (
        result.effective_temperature_information,
        max(result.determinant, 0.0),
        result.condition_number,
    )


def save_heatmap(
    data: np.ndarray,
    angles: np.ndarray,
    title: str,
    label: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.8), constrained_layout=True)
    image = ax.imshow(
        data,
        origin="lower",
        extent=[angles.min(), angles.max(), angles.min(), angles.max()],
        aspect="auto",
    )
    ax.set_xlabel(r"second angle $\theta_2$")
    ax.set_ylabel(r"first angle $\theta_1$")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output = ROOT / "results" / "schedule_optimization"
    output.mkdir(parents=True, exist_ok=True)

    parameters = CollisionParameters(
        temperature=0.8277777777777777,
        memory_angle=0.35454545454545455,
    )
    length = 8
    angles = np.concatenate(
        [np.linspace(0.18, 1.42, 9), np.asarray([0.5 * np.pi])]
    )

    fixed_effective = np.zeros(angles.size)
    fixed_determinant = np.zeros(angles.size)
    for i, angle in enumerate(angles):
        fixed_effective[i], fixed_determinant[i], _ = evaluate(
            parameters, float(angle), length
        )

    alternating_effective = np.zeros((angles.size, angles.size))
    alternating_determinant = np.zeros_like(alternating_effective)
    block_effective = np.zeros_like(alternating_effective)
    block_determinant = np.zeros_like(alternating_effective)

    total = 2 * angles.size**2
    completed = 0
    for i, first in enumerate(angles):
        for j, second in enumerate(angles):
            alternating = (float(first), float(second))
            block = tuple(
                [float(first)] * (length // 2)
                + [float(second)] * (length - length // 2)
            )
            (
                alternating_effective[i, j],
                alternating_determinant[i, j],
                _,
            ) = evaluate(parameters, alternating, length)
            completed += 1
            (
                block_effective[i, j],
                block_determinant[i, j],
                _,
            ) = evaluate(parameters, block, length)
            completed += 1
            if completed % max(total // 10, 1) == 0:
                print(f"completed {completed}/{total}")

    best_fixed_index = int(np.argmax(fixed_effective))
    best_alt_index = np.unravel_index(
        np.argmax(alternating_effective), alternating_effective.shape
    )
    best_block_index = np.unravel_index(
        np.argmax(block_effective), block_effective.shape
    )
    fixed_best = float(fixed_effective[best_fixed_index])
    fixed_det_at_best = float(fixed_determinant[best_fixed_index])

    summary = {
        "parameters": {
            "temperature": parameters.temperature,
            "memory_angle": parameters.memory_angle,
            "record_length": length,
        },
        "best_fixed": {
            "angle": float(angles[best_fixed_index]),
            "effective": fixed_best,
            "determinant": fixed_det_at_best,
        },
        "best_alternating": {
            "angles": [
                float(angles[best_alt_index[0]]),
                float(angles[best_alt_index[1]]),
            ],
            "effective": float(alternating_effective[best_alt_index]),
            "determinant": float(alternating_determinant[best_alt_index]),
            "effective_gain_over_best_fixed": float(
                alternating_effective[best_alt_index] / fixed_best
            ),
            "determinant_gain_over_best_fixed": float(
                alternating_determinant[best_alt_index] / fixed_det_at_best
            ),
        },
        "best_block": {
            "angles": [
                float(angles[best_block_index[0]]),
                float(angles[best_block_index[1]]),
            ],
            "effective": float(block_effective[best_block_index]),
            "determinant": float(block_determinant[best_block_index]),
            "effective_gain_over_best_fixed": float(
                block_effective[best_block_index] / fixed_best
            ),
            "determinant_gain_over_best_fixed": float(
                block_determinant[best_block_index] / fixed_det_at_best
            ),
        },
    }

    np.savez(
        output / "optimization_data.npz",
        angles=angles,
        fixed_effective=fixed_effective,
        fixed_determinant=fixed_determinant,
        alternating_effective=alternating_effective,
        alternating_determinant=alternating_determinant,
        block_effective=block_effective,
        block_determinant=block_determinant,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_heatmap(
        alternating_effective / fixed_best,
        angles,
        "Alternating two-setting schedule",
        "effective-information gain over best fixed",
        output / "alternating_gain.png",
    )
    save_heatmap(
        block_effective / fixed_best,
        angles,
        "Blocked two-setting schedule",
        "effective-information gain over best fixed",
        output / "block_gain.png",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
