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
from simulation.core.outputs import write_summary, write_ticket_metrics
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


def run_simulation(seed: int, horizon: float, output_dir: Path) -> None:
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
    metrics = MetricsCollector(horizon=horizon)

    config = SimulationConfig(
        arrival_rate=arrival_rate,
        feedback_review=feedback_probs["review_to_development"],
        feedback_testing=feedback_probs["testing_to_development"],
        service_params=service_params,
        horizon=horizon,
    )
    engine = SimulationEngine(config, developer_pool, rngs, metrics)
    engine.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    tickets_path = output_dir / "tickets.csv"
    summary_path = output_dir / "summary.csv"
    write_ticket_metrics(engine.state.tickets.values(), str(tickets_path))

    summary = metrics.summary()
    summary["horizon_days"] = horizon
    summary["arrivals"] = float(metrics.arrivals)
    summary["developer_count"] = float(developer_count)
    write_summary(summary, str(summary_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finite-horizon ASF simulation")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--horizon", type=float, default=365.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()

    try:
        run_simulation(args.seed, args.horizon, args.output_dir)
    except InputDataError as exc:
        raise SystemExit(f"Input error: {exc}") from exc


if __name__ == "__main__":
    main()
