# v1
# file: 8_stint_PMF.py

"""
Compute combined stint PMFs (OFF/DEV/REV/TEST) from ETL outputs and write a
single CSV table.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

from etl.state_parameters import (
    DEFAULT_PRECISION,
    build_events,
    build_transitions_and_stints,
    load_phase_durations,
    load_pr_rows,
)

DEFAULT_OUTPUT_PATH = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "stint_PMF.csv"


def write_combined_pmf(
    stints: dict[str, list[float]],
    output_path: Path,
    tolerance: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "length", "prob"])
        for state in sorted(stints.keys()):
            durations = stints[state]
            counts = Counter(durations)
            total = sum(counts.values())
            if total == 0:
                continue
            probabilities = []
            for duration in sorted(counts):
                probability = counts[duration] / total
                probabilities.append(probability)
                writer.writerow([state, f"{duration:.3f}", f"{probability:.6f}"])
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"PMF for {state} does not sum to 1.0.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined stint PMF table from PR and phase-duration ETL outputs."
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path for combined stint PMFs.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Absolute tolerance for probability checks.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal precision for rounding stint lengths (days).",
    )
    args = parser.parse_args()

    phase_durations, _ = load_phase_durations()
    pr_rows = load_pr_rows()
    developer_events, _ = build_events(phase_durations, pr_rows)
    _, stints, _ = build_transitions_and_stints(developer_events, args.precision)

    write_combined_pmf(stints, Path(args.output_path), args.tolerance)


if __name__ == "__main__":
    main()
