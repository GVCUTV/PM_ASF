import csv
import math
from statistics import NormalDist, mean, stdev
from typing import Dict, Iterable, List

from .metrics import BatchStats
from .models import Stage, Ticket


def write_ticket_metrics(tickets: Iterable[Ticket], path: str) -> None:
    fieldnames = [
        "ticket_id",
        "arrival_time",
        "completion_time",
        "time_in_system",
        "dev_cycles",
        "review_cycles",
        "testing_cycles",
        "dev_queue_wait",
        "review_queue_wait",
        "testing_queue_wait",
        "dev_service_time",
        "review_service_time",
        "testing_service_time",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ticket in tickets:
            if ticket.completion_time is None:
                continue
            row = {
                "ticket_id": ticket.ticket_id,
                "arrival_time": ticket.arrival_time,
                "completion_time": ticket.completion_time,
                "time_in_system": ticket.completion_time - ticket.arrival_time,
                "dev_cycles": ticket.stage_cycles.get(Stage.DEV, 0),
                "review_cycles": ticket.stage_cycles.get(Stage.REVIEW, 0),
                "testing_cycles": ticket.stage_cycles.get(Stage.TESTING, 0),
                "dev_queue_wait": ticket.stage_waits.get(Stage.DEV, 0.0),
                "review_queue_wait": ticket.stage_waits.get(Stage.REVIEW, 0.0),
                "testing_queue_wait": ticket.stage_waits.get(Stage.TESTING, 0.0),
                "dev_service_time": ticket.stage_services.get(Stage.DEV, 0.0),
                "review_service_time": ticket.stage_services.get(Stage.REVIEW, 0.0),
                "testing_service_time": ticket.stage_services.get(Stage.TESTING, 0.0),
            }
            writer.writerow(row)


def write_summary(summary: Dict[str, float], path: str) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def _batch_metrics(batch: BatchStats, batch_size: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for stage in Stage:
        throughput = batch.completions[stage] / batch_size if batch_size > 0 else 0.0
        utilization = (
            batch.busy_time[stage] / batch.capacity_time[stage]
            if batch.capacity_time[stage] > 0
            else 0.0
        )
        avg_queue = batch.queue_time[stage] / batch_size if batch_size > 0 else 0.0
        metrics[f"throughput_{stage.value.lower()}"] = throughput
        metrics[f"utilization_{stage.value.lower()}"] = utilization
        metrics[f"avg_queue_{stage.value.lower()}"] = avg_queue
    metrics["feedback_rate_review"] = (
        batch.feedback_review / batch.completions[Stage.REVIEW]
        if batch.completions[Stage.REVIEW] > 0
        else 0.0
    )
    metrics["feedback_rate_testing"] = (
        batch.feedback_test / batch.completions[Stage.TESTING]
        if batch.completions[Stage.TESTING] > 0
        else 0.0
    )
    metrics["closed_tickets"] = float(batch.closed_tickets)
    metrics["mean_time_in_system"] = (
        sum(batch.ticket_times) / len(batch.ticket_times) if batch.ticket_times else 0.0
    )
    return metrics


def write_batch_means(batch_stats: List[BatchStats], batch_size: float, path: str) -> List[Dict[str, float]]:
    rows = []
    for index, batch in enumerate(batch_stats, start=1):
        metrics = _batch_metrics(batch, batch_size)
        metrics["batch"] = index
        rows.append(metrics)
    fieldnames = ["batch"] + [key for key in rows[0] if key != "batch"] if rows else ["batch"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def write_confidence_intervals(batch_rows: List[Dict[str, float]], path: str) -> None:
    if not batch_rows:
        return
    metrics = [key for key in batch_rows[0] if key != "batch"]
    z = NormalDist().inv_cdf(0.975)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "ci_low", "ci_high", "batches"])
        for metric in metrics:
            values = [row[metric] for row in batch_rows]
            avg = mean(values)
            std = stdev(values) if len(values) > 1 else 0.0
            half_width = z * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            writer.writerow([metric, avg, std, avg - half_width, avg + half_width, len(values)])


def write_service_time_diagnostics(
    empirical_stats: Dict[Stage, Dict[str, float]],
    service_params: Dict[Stage, Dict[str, Dict[str, float]]],
    service_time_scale: float,
    service_time_caps: Dict[Stage, float],
    sample_counts: Dict[Stage, int],
    path: str,
) -> None:
    fieldnames = [
        "stage",
        "sampling_mode",
        "distribution",
        "parameters",
        "service_time_scale",
        "cap_days",
        "sample_count",
        "empirical_count",
        "empirical_mean_days",
        "empirical_p95_days",
        "empirical_p99_days",
        "empirical_max_days",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stage in Stage:
            params = service_params.get(stage, {})
            stats = empirical_stats.get(stage, {})
            sample_count = sample_counts.get(stage, 0)
            sampling_mode = "empirical" if sample_count > 0 else "parametric"
            writer.writerow(
                {
                    "stage": stage.value,
                    "sampling_mode": sampling_mode,
                    "distribution": params.get("distribution", ""),
                    "parameters": params.get("parameters", {}),
                    "service_time_scale": service_time_scale,
                    "cap_days": service_time_caps.get(stage),
                    "sample_count": sample_count,
                    "empirical_count": stats.get("count", 0.0),
                    "empirical_mean_days": stats.get("mean", 0.0),
                    "empirical_p95_days": stats.get("p95", 0.0),
                    "empirical_p99_days": stats.get("p99", 0.0),
                    "empirical_max_days": stats.get("max", 0.0),
                }
            )
