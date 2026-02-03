import csv
import json
import os
from typing import Dict, List, Tuple

from .models import Stage


class InputDataError(RuntimeError):
    pass


def _ensure_exists(path: str) -> str:
    if not os.path.exists(path):
        raise InputDataError(f"Missing required input file: {path}")
    return path


def load_arrival_rate(path: str) -> float:
    _ensure_exists(path)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise InputDataError(f"Arrival rate file empty: {path}")
    row = rows[0]
    for key in ("arrival_rate_per_day", "arrival_rate"):
        if key in row:
            return float(row[key])
    raise InputDataError(f"Arrival rate column not found in {path}")


def load_feedback_probabilities(path: str) -> Dict[str, float]:
    _ensure_exists(path)
    probs: Dict[str, float] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = row.get("metric") or row.get("Metric")
            probability = row.get("probability") or row.get("Probability")
            if metric is None or probability is None:
                continue
            probs[metric] = float(probability)
    if "review_to_development" not in probs or "testing_to_development" not in probs:
        raise InputDataError(f"Feedback probabilities missing expected metrics in {path}")
    return probs


def load_transition_matrix(path: str) -> Dict[str, Dict[str, float]]:
    _ensure_exists(path)
    matrix: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state = row.get("state")
            if not state:
                continue
            matrix[state] = {key: float(value) for key, value in row.items() if key != "state"}
    if not matrix:
        raise InputDataError(f"Transition matrix empty: {path}")
    return matrix


def load_stint_pmf(path: str) -> Dict[str, List[Tuple[float, float]]]:
    _ensure_exists(path)
    pmf: Dict[str, List[Tuple[float, float]]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            state = row.get("state")
            length = row.get("length")
            prob = row.get("prob")
            if state is None or length is None or prob is None:
                continue
            pmf.setdefault(state, []).append((float(length), float(prob)))
    if not pmf:
        raise InputDataError(f"Stint PMF empty: {path}")
    return pmf


def load_service_params(
    json_path: str,
    summary_path: str,
) -> Dict[Stage, Dict[str, Dict[str, float]]]:
    if os.path.exists(json_path):
        with open(json_path) as handle:
            raw = json.load(handle)
        params = {
            Stage.DEV: raw["DEV"],
            Stage.REVIEW: raw["REVIEW"],
            Stage.TESTING: raw["TESTING"],
        }
        return params

    _ensure_exists(summary_path)
    with open(summary_path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    params = {}
    for row in rows:
        phase = row.get("phase")
        if not phase:
            continue
        if phase.lower() == "dev":
            stage = Stage.DEV
        elif phase.lower() == "review":
            stage = Stage.REVIEW
        elif phase.lower() == "testing":
            stage = Stage.TESTING
        else:
            continue
        best_fit = row.get("best_fit")
        parameters = row.get("parameters")
        if not best_fit or not parameters:
            continue
        params[stage] = {"distribution": best_fit, "parameters": json.loads(parameters)}
    if not params:
        raise InputDataError("Service parameters missing in distribution summary")
    return params


def load_developer_count(initial_count_path: str, events_path: str) -> int:
    if os.path.exists(initial_count_path):
        with open(initial_count_path, newline="") as handle:
            reader = csv.DictReader(handle)
            total = 0
            for row in reader:
                count = row.get("count") or row.get("Count")
                if count is None:
                    continue
                total += int(float(count))
        if total > 0:
            return total

    if os.path.exists(events_path):
        with open(events_path, newline="") as handle:
            reader = csv.DictReader(handle)
            developers = {row.get("developer") for row in reader if row.get("developer")}
        if developers:
            return len(developers)

    raise InputDataError(
        "Developer count could not be inferred from initial_dev_count or developer_events"
    )


def load_phase_duration_stats(path: str) -> Tuple[Dict[Stage, Dict[str, float]], float]:
    if not os.path.exists(path):
        return {}, 1.0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return {}, 1.0

    stage_columns = {
        Stage.DEV: "dev_duration_hours",
        Stage.REVIEW: "review_duration_hours",
        Stage.TESTING: "testing_duration_hours",
    }
    stats: Dict[Stage, Dict[str, float]] = {}
    unit_scale = 1.0 / 24.0 if any(col in reader.fieldnames for col in stage_columns.values()) else 1.0

    for stage, column in stage_columns.items():
        values = []
        for row in rows:
            value = row.get(column)
            if value is None or value == "":
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if numeric <= 0:
                continue
            values.append(numeric * unit_scale)
        if not values:
            continue
        values.sort()
        count = len(values)
        mean = sum(values) / count
        p95_index = max(int(0.95 * count) - 1, 0)
        p99_index = max(int(0.99 * count) - 1, 0)
        stats[stage] = {
            "count": float(count),
            "mean": mean,
            "p95": values[p95_index],
            "p99": values[p99_index],
            "max": values[-1],
        }

    return stats, unit_scale
