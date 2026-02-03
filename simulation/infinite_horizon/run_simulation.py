#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.core.developer_pool import DeveloperPool
from simulation.core.engine import SimulationConfig, SimulationEngine
from simulation.core.inputs import (
    InputDataError,
    load_arrival_rate,
    load_developer_count,
    load_feedback_probabilities,
    load_service_params,
    load_stint_pmf,
    load_transition_matrix,
)
from simulation.core.metrics import MetricsCollector
from simulation.core.outputs import write_batch_means, write_confidence_intervals
from simulation.core.rng import RNGStreams


def build_paths(repo_root: Path) -> dict:
    return {
        "arrival_rate": repo_root / "etl/output/csv/arrival_rate_jira_issues.csv",
        "feedback_probs": repo_root / "etl/output/csv/feedback_probabilities.csv",
        "transition_matrix": repo_root / "etl/output/csv/transition_matrix.csv",
        "stint_pmf": repo_root / "etl/output/csv/stint_PMF.csv",
        "service_params": repo_root / "data/state_parameters/service_params.json",
        "distribution_summary": repo_root / "etl/output/csv/distribution_summary.csv",
        "initial_dev_count": repo_root / "etl/output/csv/initial_dev_count.csv",
        "developer_events": repo_root / "data/state_parameters/developer_events.csv",
    }


def run_simulation(seed: int, total_time: float, batches: int, output_dir: Path) -> None:
    paths = build_paths(REPO_ROOT)

    arrival_rate = load_arrival_rate(str(paths["arrival_rate"]))
    feedback_probs = load_feedback_probabilities(str(paths["feedback_probs"]))
    transition_matrix = load_transition_matrix(str(paths["transition_matrix"]))
    stint_pmf = load_stint_pmf(str(paths["stint_pmf"]))
    service_params = load_service_params(
        str(paths["service_params"]), str(paths["distribution_summary"])
    )
    developer_count = load_developer_count(
        str(paths["initial_dev_count"]), str(paths["developer_events"])
    )

    rngs = RNGStreams(seed)
    developer_pool = DeveloperPool(transition_matrix, stint_pmf, developer_count, rngs.developer)
    batch_size = total_time / batches
    metrics = MetricsCollector(horizon=total_time, batch_size=batch_size, batch_count=batches)

    config = SimulationConfig(
        arrival_rate=arrival_rate,
        feedback_review=feedback_probs["review_to_development"],
        feedback_testing=feedback_probs["testing_to_development"],
        service_params=service_params,
        horizon=total_time,
    )
    engine = SimulationEngine(config, developer_pool, rngs, metrics)
    engine.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_means_path = output_dir / "summary_batch_means.csv"
    ci_path = output_dir / "summary_ci.csv"
    rows = write_batch_means(metrics.batch_stats, batch_size, str(batch_means_path))
    write_confidence_intervals(rows, str(ci_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run infinite-horizon ASF simulation")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--total-time", type=float, default=3650.0)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()

    try:
        run_simulation(args.seed, args.total_time, args.batches, args.output_dir)
    except InputDataError as exc:
        raise SystemExit(f"Input error: {exc}") from exc


if __name__ == "__main__":
    main()
