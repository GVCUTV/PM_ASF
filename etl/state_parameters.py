# v1
# file: state_parameters.py

"""
Build per-developer state transition parameters (OFF/DEV/REV/TEST) and PMFs
from GitHub PR assignees and phase duration timestamps.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

PRS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "github_prs_raw.csv"
PHASE_DURATIONS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "phase_durations.csv"
DISTRIBUTION_SUMMARY_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "distribution_summary.csv"

STATES = ("OFF", "DEV", "REV", "TEST")
REQUIRED_PHASE_FIELDS = ("dev_start_ts", "review_start_ts", "review_end_ts", "testing_end_ts")
JIRA_KEY_REGEX = re.compile(r"BOOKKEEPER-\d+", re.IGNORECASE)
DEFAULT_PRECISION = 3
DEFAULT_ALPHA = 1.0


@dataclass(frozen=True)
class Event:
    state: str
    start: datetime
    end: datetime
    jira_key: str
    pr_number: str | None


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_distribution_summary() -> list[dict[str, str]]:
    if not DISTRIBUTION_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing distribution summary: {DISTRIBUTION_SUMMARY_CSV}")
    with DISTRIBUTION_SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def extract_jira_keys(*values: str | None) -> set[str]:
    keys: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in JIRA_KEY_REGEX.findall(str(value)):
            keys.add(match.upper())
    return keys


def parse_assignees(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    raw_value = raw_value.strip()
    if not raw_value or raw_value == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    assignees: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            login = item.get("login")
            if isinstance(login, str) and login.strip():
                assignees.append(login.strip())
    return assignees


def load_phase_durations() -> tuple[
    dict[str, tuple[datetime, datetime, datetime, datetime]], list[dict[str, str]]
]:
    durations: dict[str, tuple[datetime, datetime, datetime, datetime]] = {}
    skipped: list[dict[str, str]] = []
    with PHASE_DURATIONS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("jira_key") or "").strip().upper()
            if not key:
                continue
            parsed_values = [parse_timestamp(row.get(field)) for field in REQUIRED_PHASE_FIELDS]
            if any(value is None for value in parsed_values):
                skipped.append(
                    {
                        "jira_key": key,
                        "reason": row.get("exception_reason") or "missing_phase_timestamp",
                    }
                )
                continue
            dev_start, review_start, review_end, test_end = parsed_values  # type: ignore[assignment]
            if not (dev_start and review_start and review_end and test_end):
                skipped.append({"jira_key": key, "reason": "missing_phase_timestamp"})
                continue
            durations[key] = (dev_start, review_start, review_end, test_end)
    return durations, skipped


def load_pr_rows() -> list[dict[str, str]]:
    with PRS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def add_event(stints: dict[str, list[float]], event: Event, precision: int) -> bool:
    duration_days = (event.end - event.start).total_seconds() / 86400.0
    if duration_days <= 0:
        return False
    rounded = round(duration_days, precision)
    stints[event.state].append(rounded)
    return True


def add_off_gap(stints: dict[str, list[float]], gap_seconds: float, precision: int) -> bool:
    if gap_seconds <= 0:
        return False
    rounded = round(gap_seconds / 86400.0, precision)
    stints["OFF"].append(rounded)
    return True


def build_events(
    phase_durations: dict[str, tuple[datetime, datetime, datetime, datetime]],
    pr_rows: Iterable[dict[str, str]],
) -> tuple[dict[str, list[Event]], list[dict[str, str]]]:
    developer_events: dict[str, list[Event]] = defaultdict(list)
    event_rows: list[dict[str, str]] = []
    for row in pr_rows:
        assignees = parse_assignees(row.get("assignees"))
        if not assignees:
            continue
        jira_keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not jira_keys:
            continue
        pr_number = (row.get("number") or "").strip()
        for jira_key in jira_keys:
            if jira_key not in phase_durations:
                continue
            dev_start, review_start, review_end, test_end = phase_durations[jira_key]
            events = [
                Event("DEV", dev_start, review_start, jira_key=jira_key, pr_number=pr_number or None),
                Event("REV", review_start, review_end, jira_key=jira_key, pr_number=pr_number or None),
                Event("TEST", review_end, test_end, jira_key=jira_key, pr_number=pr_number or None),
            ]
            for assignee in assignees:
                developer_events[assignee].extend(events)
                for event in events:
                    event_rows.append(
                        {
                            "developer": assignee,
                            "state": event.state,
                            "start_ts": format_timestamp(event.start),
                            "end_ts": format_timestamp(event.end),
                            "jira_key": event.jira_key,
                            "pr_number": event.pr_number or "",
                        }
                    )
    return developer_events, event_rows


def build_transitions_and_stints(
    developer_events: dict[str, list[Event]],
    precision: int,
) -> tuple[dict[str, Counter[str]], dict[str, list[float]], list[dict[str, str]]]:
    transitions: dict[str, Counter[str]] = {state: Counter() for state in STATES}
    stints: dict[str, list[float]] = {state: [] for state in STATES}
    skipped: list[dict[str, str]] = []
    for developer, events in developer_events.items():
        if not events:
            continue
        events_sorted = sorted(events, key=lambda e: (e.start, e.end, e.jira_key, e.state))
        first_event = events_sorted[0]
        transitions["OFF"][first_event.state] += 1
        if not add_event(stints, first_event, precision):
            skipped.append(
                {
                    "developer": developer,
                    "jira_key": first_event.jira_key,
                    "state": first_event.state,
                    "reason": "non_positive_duration",
                }
            )
        for prev_event, next_event in zip(events_sorted, events_sorted[1:]):
            gap_seconds = (next_event.start - prev_event.end).total_seconds()
            if gap_seconds > 0:
                transitions[prev_event.state]["OFF"] += 1
                if not add_off_gap(stints, gap_seconds, precision):
                    skipped.append(
                        {
                            "developer": developer,
                            "jira_key": next_event.jira_key,
                            "state": "OFF",
                            "reason": "non_positive_gap",
                        }
                    )
                transitions["OFF"][next_event.state] += 1
            elif gap_seconds == 0:
                transitions[prev_event.state][next_event.state] += 1
            else:
                skipped.append(
                    {
                        "developer": developer,
                        "jira_key": next_event.jira_key,
                        "state": next_event.state,
                        "reason": "negative_gap",
                    }
                )
            if not add_event(stints, next_event, precision):
                skipped.append(
                    {
                        "developer": developer,
                        "jira_key": next_event.jira_key,
                        "state": next_event.state,
                        "reason": "non_positive_duration",
                    }
                )
    return transitions, stints, skipped


def write_transition_matrix(
    transitions: dict[str, Counter[str]],
    output_path: Path,
    tolerance: float,
    alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", *STATES])
        for origin in STATES:
            total = sum(transitions[origin].values())
            row: list[str | float] = [origin]
            denominator = total + alpha * len(STATES)
            probabilities = [
                (transitions[origin][dest] + alpha) / denominator for dest in STATES
            ]
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"Transition probabilities for {origin} do not sum to 1.0.")
            row.extend([round(p, 6) for p in probabilities])
            writer.writerow(row)


def write_pmf(
    stints: dict[str, list[float]],
    output_dir: Path,
    tolerance: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for state in STATES:
        durations = stints[state]
        counts = Counter(durations)
        total = sum(counts.values())
        output_path = output_dir / f"stint_PMF_{state}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["length", "prob"])
            if total == 0:
                continue
            probabilities = []
            for duration in sorted(counts):
                probability = counts[duration] / total
                probabilities.append(probability)
                writer.writerow([f"{duration:.3f}", f"{probability:.6f}"])
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"PMF for {state} does not sum to 1.0.")


def write_stint_counts(stints: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for state in STATES:
        durations = stints[state]
        counts = Counter(durations)
        output_path = output_dir / f"stint_counts_{state}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["length", "count"])
            for duration in sorted(counts):
                writer.writerow([f"{duration:.3f}", counts[duration]])


def write_transition_counts(transitions: dict[str, Counter[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin", "destination", "count"])
        for origin in STATES:
            for destination in STATES:
                writer.writerow([origin, destination, transitions[origin][destination]])


def write_event_rows(event_rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["developer", "state", "start_ts", "end_ts", "jira_key", "pr_number"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            event_rows,
            key=lambda item: (
                item["developer"],
                item["start_ts"],
                item["end_ts"],
                item["jira_key"],
                item["state"],
            ),
        ):
            writer.writerow(row)


def write_key_map(pr_rows: Iterable[dict[str, str]], output_path: Path) -> list[dict[str, str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for row in pr_rows:
        jira_keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not jira_keys:
            continue
        pr_number = (row.get("number") or "").strip()
        pr_url = (row.get("html_url") or "").strip()
        for jira_key in jira_keys:
            rows.append(
                {
                    "jira_key": jira_key,
                    "pr_number": pr_number,
                    "pr_url": pr_url,
                }
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["jira_key", "pr_number", "pr_url"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (item["jira_key"], item["pr_number"])))
    return rows


def write_developer_ticket_map(
    pr_rows: Iterable[dict[str, str]], output_path: Path
) -> list[dict[str, str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for row in pr_rows:
        assignees = parse_assignees(row.get("assignees"))
        if not assignees:
            continue
        jira_keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not jira_keys:
            continue
        pr_number = (row.get("number") or "").strip()
        for jira_key in jira_keys:
            for assignee in assignees:
                rows.append(
                    {
                        "developer": assignee,
                        "jira_key": jira_key,
                        "pr_number": pr_number,
                    }
                )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["developer", "jira_key", "pr_number"])
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda item: (item["developer"], item["jira_key"], item["pr_number"]))
        )
    return rows


def write_skipped_rows(
    rows: list[dict[str, str]],
    output_path: Path,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_service_params(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params: dict[str, dict[str, object]] = {}
    for row in summary_rows:
        phase = (row.get("phase") or "").strip().lower()
        if phase not in {"dev", "review", "testing"}:
            continue
        best_fit = (row.get("best_fit") or "").strip()
        raw_params = row.get("parameters") or "{}"
        try:
            parsed_params = json.loads(raw_params)
        except json.JSONDecodeError:
            parsed_params = {}
        params[phase.upper()] = {
            "distribution": best_fit,
            "parameters": parsed_params,
        }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build state transition parameters from PR data.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(PROJECT_ROOT) / "data" / "state_parameters"),
        help="Directory for transition matrix and PMF CSVs.",
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Laplace smoothing constant for transition probabilities.",
    )
    args = parser.parse_args()

    summary_rows = load_distribution_summary()
    phase_durations, skipped_phases = load_phase_durations()
    pr_rows = load_pr_rows()
    developer_events, event_rows = build_events(phase_durations, pr_rows)
    transitions, stints, skipped_events = build_transitions_and_stints(
        developer_events, args.precision
    )

    output_dir = Path(args.output_dir)
    write_key_map(pr_rows, output_dir / "jira_pr_key_map.csv")
    write_developer_ticket_map(pr_rows, output_dir / "developer_ticket_map.csv")
    write_event_rows(event_rows, output_dir / "developer_events.csv")
    write_transition_counts(transitions, output_dir / "transition_counts.csv")
    write_transition_matrix(
        transitions,
        output_dir / "matrix_P.csv",
        args.tolerance,
        args.alpha,
    )
    write_pmf(stints, output_dir, args.tolerance)
    write_stint_counts(stints, output_dir)
    write_service_params(summary_rows, output_dir / "service_params.json")
    write_skipped_rows(
        skipped_phases,
        output_dir / "skipped_phase_rows.csv",
        ["jira_key", "reason"],
    )
    write_skipped_rows(
        skipped_events,
        output_dir / "skipped_event_rows.csv",
        ["developer", "jira_key", "state", "reason"],
    )


if __name__ == "__main__":
    main()
