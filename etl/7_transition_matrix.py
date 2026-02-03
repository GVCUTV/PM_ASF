# v1
# file: 7_transition_matrix.py

"""
Generate the OFF/DEV/REV/TEST transition matrix from ETL output CSVs.
"""

from __future__ import annotations

import argparse
import ast
import csv
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

STATES = ("OFF", "DEV", "REV", "TEST")
REQUIRED_PHASE_FIELDS = ("dev_start_ts", "review_start_ts", "review_end_ts", "testing_end_ts")
JIRA_KEY_REGEX = re.compile(r"BOOKKEEPER-\d+", re.IGNORECASE)
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


def load_phase_durations() -> dict[str, tuple[datetime, datetime, datetime, datetime]]:
    durations: dict[str, tuple[datetime, datetime, datetime, datetime]] = {}
    with PHASE_DURATIONS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("jira_key") or "").strip().upper()
            if not key:
                continue
            parsed_values = [parse_timestamp(row.get(field)) for field in REQUIRED_PHASE_FIELDS]
            if any(value is None for value in parsed_values):
                continue
            dev_start, review_start, review_end, test_end = parsed_values  # type: ignore[assignment]
            if not (dev_start and review_start and review_end and test_end):
                continue
            durations[key] = (dev_start, review_start, review_end, test_end)
    return durations


def load_pr_rows() -> list[dict[str, str]]:
    with PRS_CSV.open(newline="", encoding="utf-8") as handle:
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


def build_events(
    phase_durations: dict[str, tuple[datetime, datetime, datetime, datetime]],
    pr_rows: Iterable[dict[str, str]],
) -> dict[str, list[Event]]:
    developer_events: dict[str, list[Event]] = defaultdict(list)
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
    return developer_events


def build_transitions(
    developer_events: dict[str, list[Event]],
) -> dict[str, Counter[str]]:
    transitions: dict[str, Counter[str]] = {state: Counter() for state in STATES}
    for events in developer_events.values():
        if not events:
            continue
        events_sorted = sorted(events, key=lambda e: (e.start, e.end, e.jira_key, e.state))
        first_event = events_sorted[0]
        transitions["OFF"][first_event.state] += 1
        for prev_event, next_event in zip(events_sorted, events_sorted[1:]):
            gap_seconds = (next_event.start - prev_event.end).total_seconds()
            if gap_seconds > 0:
                transitions[prev_event.state]["OFF"] += 1
                transitions["OFF"][next_event.state] += 1
            elif gap_seconds == 0:
                transitions[prev_event.state][next_event.state] += 1
    return transitions


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
            denominator = total + alpha * len(STATES)
            probabilities = [
                (transitions[origin][dest] + alpha) / denominator for dest in STATES
            ]
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"Transition probabilities for {origin} do not sum to 1.0.")
            writer.writerow([origin, *[f"{p:.6f}" for p in probabilities]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OFF/DEV/REV/TEST transition matrix.")
    parser.add_argument(
        "--output-path",
        default=str(Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "transition_matrix.csv"),
        help="CSV path for transition matrix output.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Absolute tolerance for probability checks.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Laplace smoothing constant for transition probabilities.",
    )
    args = parser.parse_args()

    phase_durations = load_phase_durations()
    pr_rows = load_pr_rows()
    developer_events = build_events(phase_durations, pr_rows)
    transitions = build_transitions(developer_events)
    write_transition_matrix(
        transitions,
        Path(args.output_path),
        args.tolerance,
        args.alpha,
    )


if __name__ == "__main__":
    main()
