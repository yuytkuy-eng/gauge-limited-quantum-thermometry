"""Estimate Fisher-information rates from an ensemble of long trajectories."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.model import (
    CollisionParameters,
    record_log_likelihood,
    sample_record,
)


def likelihood_score(
    parameters: CollisionParameters,
    probe_angle: float,
    record: np.ndarray,
    temperature_step: float = 1e-3,
    memory_step: float = 1e-3,
) -> np.ndarray:
    """Central-difference score of one record likelihood."""

    plus_temperature = record_log_likelihood(
        replace(parameters, temperature=parameters.temperature + temperature_step),
        probe_angle,
        record,
    )
    minus_temperature = record_log_likelihood(
        replace(parameters, temperature=parameters.temperature - temperature_step),
        probe_angle,
        record,
    )
    plus_memory = record_log_likelihood(
        replace(parameters, memory_angle=parameters.memory_angle + memory_step),
        probe_angle,
        record,
    )
    minus_memory = record_log_likelihood(
        replace(parameters, memory_angle=parameters.memory_angle - memory_step),
        probe_angle,
        record,
    )
    return np.array(
        [
            (plus_temperature - minus_temperature) / (2.0 * temperature_step),
            (plus_memory - minus_memory) / (2.0 * memory_step),
        ]
    )


def effective_rate_from_scores(scores: np.ndarray, length: int) -> float:
    """Schur-complement temperature rate from a trajectory-score ensemble."""

    information = np.cov(scores, rowvar=False, ddof=1) / length
    effective = information[0, 0] - information[0, 1] ** 2 / information[1, 1]
    return float(max(effective, 0.0))


def bootstrap_effective_rates(
    scores: np.ndarray,
    length: int,
    rng: np.random.Generator,
    samples: int = 5000,
) -> np.ndarray:
    """Nonparametric trajectory bootstrap for the effective information rate."""

    trajectory_count = scores.shape[0]
    bootstrapped = np.zeros(samples)
    for index in range(samples):
        selection = rng.integers(0, trajectory_count, size=trajectory_count)
        bootstrapped[index] = effective_rate_from_scores(
            scores[selection], length
        )
    return bootstrapped


def main() -> None:
    output = ROOT / "results" / "information_rate"
    output.mkdir(parents=True, exist_ok=True)

    parameters = CollisionParameters(
        temperature=0.8277777777777777,
        memory_angle=0.35454545454545455,
        system_memory_angle=0.55,
    )
    protocols = {
        "weak": 0.72,
        "full_swap": 0.5 * np.pi,
    }
    lengths = (500, 2000)
    trajectory_count = 32
    bootstrap_samples = 5000
    rows: list[dict[str, float | int | str]] = []
    all_scores: dict[tuple[str, int], np.ndarray] = {}

    for protocol_index, (name, angle) in enumerate(protocols.items()):
        scores = {length: np.zeros((trajectory_count, 2)) for length in lengths}
        record_means = {length: np.zeros(trajectory_count) for length in lengths}
        for trajectory in range(trajectory_count):
            rng = np.random.default_rng(18000 + 1000 * protocol_index + trajectory)
            full_record = sample_record(parameters, angle, max(lengths), rng)
            for length in lengths:
                prefix = full_record[:length]
                scores[length][trajectory] = likelihood_score(
                    parameters, angle, prefix
                )
                record_means[length][trajectory] = prefix.mean()
            if (trajectory + 1) % 8 == 0:
                print(f"{name}: completed {trajectory + 1}/{trajectory_count}")

        for length in lengths:
            all_scores[(name, length)] = scores[length].copy()
            # The score has zero expectation at the true parameter. The sample
            # covariance is a positive-semidefinite, finite-ensemble estimate
            # of E[s s^T], with centering suppressing trajectory imbalance.
            information = np.cov(scores[length], rowvar=False, ddof=1)
            rate = information / length
            eigenvalues = np.linalg.eigvalsh(rate)
            effective_rate = rate[0, 0] - rate[0, 1] ** 2 / rate[1, 1]
            condition = float(eigenvalues[1] / eigenvalues[0])
            bootstrap = bootstrap_effective_rates(
                scores[length],
                length,
                np.random.default_rng(44000 + 100 * protocol_index + length),
                samples=bootstrap_samples,
            )
            rows.append(
                {
                    "protocol": name,
                    "probe_angle": angle,
                    "length": length,
                    "trajectory_count": trajectory_count,
                    "rate_tt": float(rate[0, 0]),
                    "rate_tm": float(rate[0, 1]),
                    "rate_mm": float(rate[1, 1]),
                    "effective_temperature_rate": float(effective_rate),
                    "effective_rate_bootstrap_low": float(
                        np.quantile(bootstrap, 0.025)
                    ),
                    "effective_rate_bootstrap_high": float(
                        np.quantile(bootstrap, 0.975)
                    ),
                    "minimum_eigenvalue_rate": float(eigenvalues[0]),
                    "maximum_eigenvalue_rate": float(eigenvalues[1]),
                    "condition_number": condition,
                    "mean_record_excitation": float(record_means[length].mean()),
                    "mean_score_temperature": float(scores[length][:, 0].mean()),
                    "mean_score_memory": float(scores[length][:, 1].mean()),
                }
            )

    with (output / "information_rate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        output / "trajectory_scores.npz",
        **{
            f"{name}_N{length}": score_values
            for (name, length), score_values in all_scores.items()
        },
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for name in protocols:
        selected = [row for row in rows if row["protocol"] == name]
        ax.plot(
            [int(row["length"]) for row in selected],
            [float(row["effective_temperature_rate"]) for row in selected],
            "o-",
            label=name.replace("_", " "),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("record length N")
    ax.set_ylabel(r"score-covariance rate $I_{T|\mu}/N$")
    ax.set_title("Long-record identifiability depends on readout strength")
    ax.legend()
    fig.savefig(output / "effective_information_rate.png", dpi=180)
    plt.close(fig)

    final_rows = {
        str(row["protocol"]): row
        for row in rows
        if int(row["length"]) == max(lengths)
    }
    weak_rate = float(final_rows["weak"]["effective_temperature_rate"])
    strong_rate = float(
        final_rows["full_swap"]["effective_temperature_rate"]
    )
    bootstrap_rng = np.random.default_rng(77531)
    weak_bootstrap = bootstrap_effective_rates(
        all_scores[("weak", max(lengths))],
        max(lengths),
        bootstrap_rng,
        bootstrap_samples,
    )
    strong_bootstrap = bootstrap_effective_rates(
        all_scores[("full_swap", max(lengths))],
        max(lengths),
        bootstrap_rng,
        bootstrap_samples,
    )
    ratio_bootstrap = np.divide(
        strong_bootstrap,
        weak_bootstrap,
        out=np.full_like(strong_bootstrap, np.nan),
        where=weak_bootstrap > 0.0,
    )
    summary = {
        "parameters": {
            "temperature": parameters.temperature,
            "memory_angle": parameters.memory_angle,
            "system_memory_angle": parameters.system_memory_angle,
        },
        "trajectory_count": trajectory_count,
        "bootstrap_samples": bootstrap_samples,
        "maximum_record_length": max(lengths),
        "weak_effective_rate_at_max_N": weak_rate,
        "full_swap_effective_rate_at_max_N": strong_rate,
        "full_swap_to_weak_rate_ratio": strong_rate / weak_rate,
        "full_swap_to_weak_ratio_bootstrap_95_interval": [
            float(np.nanquantile(ratio_bootstrap, 0.025)),
            float(np.nanquantile(ratio_bootstrap, 0.975)),
        ],
        "final_rows": final_rows,
        "caution": "Percentile trajectory bootstrap from 32 paths; larger ensembles remain desirable.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
