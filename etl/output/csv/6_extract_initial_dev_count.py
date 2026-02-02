# v1
# file: 6_extract_initial_dev_count.py

"""
Extract the initial developer counts in DEV/REV/TEST stages from raw Jira/GitHub exports.
"""

from __future__ import annotations

import ast
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

JIRA_ISSUES_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_issues_raw.csv"
PRS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "github_prs_raw.csv"
OUTPUT_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "initial_dev_count.csv"

JIRA_KEY_REGEX = re.compile(r"BOOKKEEPER-\d+", re.IGNORECASE)
STAGES = ("DEV", "REV", "TEST")
STAGE_PRIORITY = {"DEV": 0, "REV": 1, "TEST": 2}


@dataclass(frozen=True)
class PhaseBounds:
    dev_start: datetime | None
    review_start: datetime | None
    review_end: datetime | None
    testing_end: datetime | None


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
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


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


def load_jira_rows() -> list[dict[str, str]]:
    if not JIRA_ISSUES_CSV.exists():
        raise FileNotFoundError(f"Missing Jira raw CSV export: {JIRA_ISSUES_CSV}")
    with JIRA_ISSUES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_pr_rows() -> list[dict[str, str]]:
    if not PRS_CSV.exists():
        raise FileNotFoundError(f"Missing GitHub PR raw CSV export: {PRS_CSV}")
    with PRS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_pr_index(pr_rows: list[dict[str, str]]) -> dict[str, dict[str, datetime | None]]:
    pr_index: dict[str, dict[str, datetime | None]] = {}
    for row in pr_rows:
        keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not keys:
            continue
        created = parse_timestamp(row.get("created_at"))
        merged = parse_timestamp(row.get("merged_at"))
        closed = parse_timestamp(row.get("closed_at"))
        for key in keys:
            stats = pr_index.setdefault(
                key,
                {"first_created": None, "last_merged": None, "last_closed": None},
            )
            if created and (stats["first_created"] is None or created < stats["first_created"]):
                stats["first_created"] = created
            if merged and (stats["last_merged"] is None or merged > stats["last_merged"]):
                stats["last_merged"] = merged
            if closed and (stats["last_closed"] is None or closed > stats["last_closed"]):
                stats["last_closed"] = closed
    return pr_index


def build_jira_index(jira_rows: list[dict[str, str]]) -> dict[str, PhaseBounds]:
    jira_index: dict[str, PhaseBounds] = {}
    for row in jira_rows:
        key = (row.get("key") or "").strip().upper()
        if not key:
            continue
        created = parse_timestamp(row.get("fields.created") or row.get("created"))
        resolved = parse_timestamp(row.get("fields.resolutiondate") or row.get("resolutiondate"))
        jira_index[key] = PhaseBounds(
            dev_start=created,
            review_start=None,
            review_end=None,
            testing_end=resolved,
        )
    return jira_index


def build_phase_bounds(
    jira_index: dict[str, PhaseBounds],
    pr_index: dict[str, dict[str, datetime | None]],
) -> dict[str, PhaseBounds]:
    phases: dict[str, PhaseBounds] = {}
    for key, jira_phase in jira_index.items():
        pr_stats = pr_index.get(key, {})
        review_start = pr_stats.get("first_created") if pr_stats else None
        review_end = pr_stats.get("last_merged") or pr_stats.get("last_closed") if pr_stats else None
        testing_end = jira_phase.testing_end or review_end
        phases[key] = PhaseBounds(
            dev_start=jira_phase.dev_start,
            review_start=review_start,
            review_end=review_end,
            testing_end=testing_end,
        )
    return phases


def build_assignee_map(pr_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    assignee_map: dict[str, set[str]] = defaultdict(set)
    for row in pr_rows:
        assignees = parse_assignees(row.get("assignees"))
        if not assignees:
            continue
        jira_keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not jira_keys:
            continue
        for key in jira_keys:
            assignee_map[key].update(assignees)
    return assignee_map


def add_stage_event(
    developer_events: dict[str, list[tuple[datetime, str]]],
    developer: str,
    stage: str,
    start: datetime | None,
    end: datetime | None,
) -> None:
    if start is None or end is None:
        return
    if end < start:
        return
    developer_events[developer].append((start, stage))


def compute_initial_counts(
    phases: dict[str, PhaseBounds],
    assignee_map: dict[str, set[str]],
) -> dict[str, int]:
    developer_events: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for key, developers in assignee_map.items():
        bounds = phases.get(key)
        if not bounds:
            continue
        for developer in developers:
            add_stage_event(developer_events, developer, "DEV", bounds.dev_start, bounds.review_start)
            add_stage_event(
                developer_events,
                developer,
                "REV",
                bounds.review_start,
                bounds.review_end,
            )
            add_stage_event(
                developer_events,
                developer,
                "TEST",
                bounds.review_end,
                bounds.testing_end,
            )

    counts = {stage: 0 for stage in STAGES}
    for events in developer_events.values():
        if not events:
            continue
        first_start, first_stage = min(
            events,
            key=lambda item: (item[0], STAGE_PRIORITY.get(item[1], 99)),
        )
        if first_stage in counts:
            counts[first_stage] += 1
    return counts


def write_output(counts: dict[str, int]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "developer_count"])
        for stage in STAGES:
            writer.writerow([stage, counts.get(stage, 0)])


def main() -> None:
    jira_rows = load_jira_rows()
    pr_rows = load_pr_rows()
    pr_index = build_pr_index(pr_rows)
    jira_index = build_jira_index(jira_rows)
    phases = build_phase_bounds(jira_index, pr_index)
    assignee_map = build_assignee_map(pr_rows)
    counts = compute_initial_counts(phases, assignee_map)
    write_output(counts)


if __name__ == "__main__":
    main()
