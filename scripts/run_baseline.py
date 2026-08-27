"""Run the first exact Fisher-information stop/go scan."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.fisher import classical_fisher_matrix
from memory_thermometry.model import CollisionParameters


def scan_protocol(
    temperatures: np.ndarray,
    memories: np.ndarray,
    probe_angles: float | tuple[float, float],
    record_length: int,
) -> dict[str, np.ndarray]:
    shape = (temperatures.size, memories.size)
    naive = np.zeros(shape)
    effective = np.zeros(shape)
    determinant = np.zeros(shape)
    condition = np.zeros(shape)
    cross = np.zeros(shape)

    total = temperatures.size * memories.size
    completed = 0
    for i, temperature in enumerate(temperatures):
        for j, memory in enumerate(memories):
            parameters = CollisionParameters(
                temperature=float(temperature),
                memory_angle=float(memory),
            )
            result = classical_fisher_matrix(
                parameters,
                probe_angles=probe_angles,
                length=record_length,
            )
            naive[i, j] = result.matrix[0, 0]
            effective[i, j] = result.effective_temperature_information
            determinant[i, j] = max(result.determinant, 0.0)
            condition[i, j] = result.condition_number
            cross[i, j] = result.matrix[0, 1]
            completed += 1
            if completed % max(total // 10, 1) == 0:
                print(f"  completed {completed}/{total}")

    return {
        "naive": naive,
        "effective": effective,
        "determinant": determinant,
        "condition": condition,
        "cross": cross,
    }


def heatmap(
    data: np.ndarray,
    temperatures: np.ndarray,
    memories: np.ndarray,
    title: str,
    colorbar_label: str,
    path: Path,
    center_at_one: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    finite = data[np.isfinite(data)]
    kwargs: dict[str, object] = {}
    if center_at_one and finite.size:
        distance = max(abs(finite.min() - 1.0), abs(finite.max() - 1.0))
        kwargs.update(vmin=1.0 - distance, vmax=1.0 + distance, cmap="coolwarm")
    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[memories.min(), memories.max(), temperatures.min(), temperatures.max()],
        **kwargs,
    )
    ax.set_xlabel(r"memory angle $\mu$")
    ax.set_ylabel(r"temperature $T/\omega$")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output = ROOT / "results" / "baseline"
    output.mkdir(parents=True, exist_ok=True)

    temperatures = np.linspace(0.35, 2.50, 10)
    memories = np.linspace(0.0, 1.30, 12)
    record_length = 8

    protocols: dict[str, float | tuple[float, float]] = {
        "fixed": 0.72,
        "alternating": (0.42, 1.02),
    }
    all_results: dict[str, dict[str, np.ndarray]] = {}
    for name, angles in protocols.items():
        print(f"Scanning {name} protocol with angles={angles}")
        all_results[name] = scan_protocol(
            temperatures, memories, angles, record_length
        )

    fixed = all_results["fixed"]
    alternating = all_results["alternating"]

    fixed_markov = fixed["naive"][:, [0]]
    apparent_ratio = np.divide(
        fixed["naive"], fixed_markov, out=np.ones_like(fixed["naive"]), where=fixed_markov > 0
    )
    nuisance_ratio = np.divide(
        fixed["effective"], fixed_markov, out=np.ones_like(fixed["effective"]), where=fixed_markov > 0
    )
    alternating_recovery = np.divide(
        alternating["effective"],
        fixed["effective"],
        out=np.ones_like(fixed["effective"]),
        where=fixed["effective"] > 1e-14,
    )
    determinant_gain = np.divide(
        alternating["determinant"],
        fixed["determinant"],
        out=np.ones_like(fixed["determinant"]),
        where=fixed["determinant"] > 1e-18,
    )

    reversal_mask = (apparent_ratio > 1.0) & (nuisance_ratio < 1.0)
    recovery_mask = (alternating_recovery >= 1.30) & (determinant_gain >= 5.0)

    np.savez(
        output / "scan_data.npz",
        temperatures=temperatures,
        memories=memories,
        apparent_ratio=apparent_ratio,
        nuisance_ratio=nuisance_ratio,
        alternating_recovery=alternating_recovery,
        determinant_gain=determinant_gain,
        fixed_naive=fixed["naive"],
        fixed_effective=fixed["effective"],
        alternating_naive=alternating["naive"],
        alternating_effective=alternating["effective"],
    )

    heatmap(
        apparent_ratio,
        temperatures,
        memories,
        "Apparent memory advantage (memory assumed known)",
        r"$I_{TT}(\mu)/I_{TT}(0)$",
        output / "apparent_memory_advantage.png",
        center_at_one=True,
    )
    heatmap(
        nuisance_ratio,
        temperatures,
        memories,
        "Temperature information with unknown memory",
        r"$I_{T|\mu}(\mu)/I_{TT}(0)$",
        output / "nuisance_corrected_information.png",
        center_at_one=True,
    )
    heatmap(
        alternating_recovery,
        temperatures,
        memories,
        "Alternating-angle recovery",
        r"$I_{T|\mu}^{alt}/I_{T|\mu}^{fixed}$",
        output / "alternating_recovery.png",
        center_at_one=True,
    )

    best_recovery_index = np.unravel_index(
        np.nanargmax(alternating_recovery), alternating_recovery.shape
    )
    best_det_index = np.unravel_index(
        np.nanargmax(determinant_gain), determinant_gain.shape
    )
    summary = {
        "record_length": record_length,
        "system_memory_angle": CollisionParameters(1.0, 0.0).system_memory_angle,
        "protocols": {key: value for key, value in protocols.items()},
        "reversal_points": int(reversal_mask.sum()),
        "recovery_points_meeting_threshold": int(recovery_mask.sum()),
        "max_apparent_ratio": float(np.nanmax(apparent_ratio)),
        "min_nuisance_ratio": float(np.nanmin(nuisance_ratio[:, 1:])),
        "max_alternating_recovery": float(np.nanmax(alternating_recovery)),
        "best_recovery_at": {
            "temperature": float(temperatures[best_recovery_index[0]]),
            "memory_angle": float(memories[best_recovery_index[1]]),
        },
        "max_determinant_gain": float(np.nanmax(determinant_gain)),
        "best_determinant_gain_at": {
            "temperature": float(temperatures[best_det_index[0]]),
            "memory_angle": float(memories[best_det_index[1]]),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (output / "scan_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "temperature",
                "memory_angle",
                "apparent_ratio",
                "nuisance_ratio",
                "alternating_recovery",
                "determinant_gain",
            ]
        )
        for i, temperature in enumerate(temperatures):
            for j, memory in enumerate(memories):
                writer.writerow(
                    [
                        temperature,
                        memory,
                        apparent_ratio[i, j],
                        nuisance_ratio[i, j],
                        alternating_recovery[i, j],
                        determinant_gain[i, j],
                    ]
                )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

