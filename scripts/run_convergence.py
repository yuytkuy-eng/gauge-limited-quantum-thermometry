"""Check finite-difference and record-length convergence at a reversal point."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_thermometry.fisher import classical_fisher_matrix
from memory_thermometry.model import CollisionParameters


def main() -> None:
    output = ROOT / "results" / "convergence"
    output.mkdir(parents=True, exist_ok=True)

    parameters = CollisionParameters(
        temperature=0.8277777777777777,
        memory_angle=0.35454545454545455,
    )
    probe_angle = 0.72
    lengths = (4, 6, 8, 10, 12, 14)
    steps = (1e-3, 5e-4, 1e-4, 5e-5)

    # Vary the finite-difference step at N=8, then vary N at one converged step.
    # This avoids an unnecessary Cartesian product whose N=14 cases dominate
    # the exact-enumeration runtime.
    configurations = [(8, step) for step in steps]
    configurations.extend((length, 1e-4) for length in lengths if length != 8)

    rows: list[dict[str, float | int]] = []
    for length, step in configurations:
        result = classical_fisher_matrix(
            parameters,
            probe_angles=probe_angle,
            length=length,
            temperature_step=step,
            memory_step=step,
        )
        rows.append(
            {
                "length": length,
                "step": step,
                "i_tt": float(result.matrix[0, 0]),
                "i_tm": float(result.matrix[0, 1]),
                "i_mm": float(result.matrix[1, 1]),
                "effective": result.effective_temperature_information,
                "determinant": result.determinant,
                "condition_number": result.condition_number,
            }
        )
        print(
            f"N={length:2d}, h={step:.0e}, "
            f"I_TT={result.matrix[0, 0]:.8f}, "
            f"I_eff={result.effective_temperature_information:.8f}"
        )

    with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reference_rows = sorted(
        (row for row in rows if row["step"] == 1e-4),
        key=lambda row: int(row["length"]),
    )
    n = np.asarray([row["length"] for row in reference_rows])
    naive = np.asarray([row["i_tt"] for row in reference_rows])
    effective = np.asarray([row["effective"] for row in reference_rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(n, naive, "o-", label=r"naive $I_{TT}$")
    ax.plot(n, effective, "s-", label=r"effective $I_{T|\mu}$")
    ax.set_xlabel("record length N")
    ax.set_ylabel("Fisher information")
    ax.set_title("Fixed readout: nuisance-limited temperature information")
    ax.legend()
    fig.savefig(output / "record_length_scaling.png", dpi=180)
    plt.close(fig)

    length_eight = [row for row in rows if row["length"] == 8]
    reference = next(row for row in length_eight if row["step"] == 5e-5)
    relative_spread = max(
        abs(float(row["effective"]) - float(reference["effective"]))
        / float(reference["effective"])
        for row in length_eight
    )
    summary = {
        "parameters": {
            "temperature": parameters.temperature,
            "memory_angle": parameters.memory_angle,
            "probe_angle": probe_angle,
        },
        "relative_effective_information_spread_at_N8": relative_spread,
        "naive_growth_N4_to_N14": float(naive[-1] / naive[0]),
        "effective_growth_N4_to_N14": float(effective[-1] / effective[0]),
        "condition_number_at_N14": float(reference_rows[-1]["condition_number"]),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
