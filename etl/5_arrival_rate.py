# v1
# file: 5_arrival_rate.py

"""
Compute the Jira issue arrival rate per day from the raw Jira CSV export.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

JIRA_ISSUES_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_issues_raw.csv"
JIRA_TICKETS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_tickets_raw.csv"
OUTPUT_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "arrival_rate_jira_issues.csv"


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


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_jira_csv() -> Path:
    if JIRA_ISSUES_CSV.exists():
        return JIRA_ISSUES_CSV
    if JIRA_TICKETS_CSV.exists():
        return JIRA_TICKETS_CSV
    raise FileNotFoundError("Missing Jira raw CSV export (jira_issues_raw.csv).")


def load_created_timestamps(path: Path) -> list[datetime]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        timestamps: list[datetime] = []
        for row in reader:
            value = row.get("fields.created") or row.get("created")
            parsed = parse_timestamp(value)
            if parsed is not None:
                timestamps.append(parsed)
    return timestamps


def compute_arrival_rate(timestamps: list[datetime]) -> dict[str, str]:
    if len(timestamps) < 2:
        raise ValueError("Need at least two Jira issues with valid created timestamps.")
    timestamps.sort()
    start = timestamps[0]
    end = timestamps[-1]
    span_seconds = (end - start).total_seconds()
    if span_seconds <= 0:
        raise ValueError("Non-positive timespan for Jira issue arrivals.")
    span_days = span_seconds / 86400.0
    rate_per_day = len(timestamps) / span_days
    return {
        "ticket_count": str(len(timestamps)),
        "start_timestamp": format_timestamp(start),
        "end_timestamp": format_timestamp(end),
        "span_days": f"{span_days:.6f}",
        "arrival_rate_per_day": f"{rate_per_day:.6f}",
    }


def write_output(row: dict[str, str]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    jira_path = resolve_jira_csv()
    timestamps = load_created_timestamps(jira_path)
    result = compute_arrival_rate(timestamps)
    write_output(result)


if __name__ == "__main__":
    main()
