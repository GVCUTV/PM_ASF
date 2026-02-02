# v1
# file: etl/4_feedback_probabilities_etl.py
"""
Extract PR-level feedback probabilities for:
- Review -> Development (changes requested during review)
- Testing -> Development (CI/check failures during testing)
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

PRS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "github_prs_raw.csv"
JIRA_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_issues_raw.csv"
OUTPUT_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "feedback_probabilities.csv"
ENRICHED_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "feedback_enriched.csv"
LOG_PATH = Path(PROJECT_ROOT) / "etl" / "output" / "logs" / "feedback_probabilities.log"

CREATED_COL_CANDIDATES = ("created_at", "created", "fields.created")

REVIEW_COUNT_COLUMN = "requested_changes_count"
REVIEW_STATES_COLUMN = "pull_request_review_states"
TEST_CONCLUSIONS_COLUMN = "check_runs_conclusions"



def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_PATH),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


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
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def to_utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _to_listish(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw or raw.lower() in {"none", "null", "nan", "[]"}:
        return []
    if raw.startswith("[") or raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    tokens = re.split(r"[\s,;|]+", raw)
    return [token.strip() for token in tokens if token.strip()]




def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def review_feedback_from_row(row: dict[str, str]) -> bool:
    requested_changes = _parse_float(row.get(REVIEW_COUNT_COLUMN)) or 0.0
    if requested_changes > 0:
        return True
    review_states = _to_listish(row.get(REVIEW_STATES_COLUMN))
    return any(token.strip().lower() == "changes_requested" for token in review_states)


def testing_feedback_from_row(row: dict[str, str]) -> bool:
    check_runs = _to_listish(row.get(TEST_CONCLUSIONS_COLUMN))
    return any(token.strip().lower() == "failure" for token in check_runs)


def load_pr_rows() -> tuple[list[dict[str, str]], list[str]]:
    if not PRS_CSV.exists():
        raise FileNotFoundError(f"Missing PR CSV: {PRS_CSV}")
    with PRS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    return rows, header


def load_jira_rows() -> list[dict[str, str]]:
    if not JIRA_CSV.exists():
        return []
    with JIRA_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def ensure_required_signal_columns(header: list[str]) -> None:
    missing = []
    if REVIEW_COUNT_COLUMN not in header and REVIEW_STATES_COLUMN not in header:
        missing.append(f"{REVIEW_COUNT_COLUMN} or {REVIEW_STATES_COLUMN}")
    if TEST_CONCLUSIONS_COLUMN not in header:
        missing.append(TEST_CONCLUSIONS_COLUMN)
    if missing:
        raise ValueError(
            "Missing feedback signal columns in raw data. "
            f"Expected: {', '.join(missing)}."
        )


def enrich_pr_rows(pr_rows: list[dict[str, str]], header: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    return pr_rows, []


def compute_feedback_probabilities(
    pr_rows: list[dict[str, str]],
    created_col: str,
    start_str: str | None,
    end_str: str | None,
) -> tuple[dict[str, dict[str, float | int]], dict[str, list[str]]]:
    start = to_utc_naive(parse_timestamp(start_str)) if start_str else None
    end = to_utc_naive(parse_timestamp(end_str)) if end_str else None
    if (start_str and start is None) or (end_str and end is None):
        raise ValueError("Invalid start/end timestamp. Use ISO8601 timestamps.")
    if start is not None and end is not None and start >= end:
        raise ValueError("Start timestamp must be before end timestamp.")

    created_values: list[datetime] = []
    missing_created = 0
    for row in pr_rows:
        created_raw = row.get(created_col)
        created_ts = to_utc_naive(parse_timestamp(created_raw))
        if created_ts is None:
            missing_created += 1
            continue
        created_values.append(created_ts)
    if not created_values:
        raise ValueError(f"No valid '{created_col}' timestamps found in input data.")

    start = start if start is not None else min(created_values)
    if end is None:
        end = max(created_values) + timedelta(microseconds=1)

    filtered_rows: list[dict[str, str]] = []
    for row in pr_rows:
        created_raw = row.get(created_col)
        created_ts = to_utc_naive(parse_timestamp(created_raw))
        if created_ts is None:
            continue
        if start <= created_ts < end:
            filtered_rows.append(row)
    if not filtered_rows:
        raise ValueError(
            f"No rows found in time window [{start.isoformat()}, {end.isoformat()}) "
            f"using created column '{created_col}'."
        )

    review_feedback_total = sum(1 for row in filtered_rows if review_feedback_from_row(row))
    testing_feedback_total = sum(1 for row in filtered_rows if testing_feedback_from_row(row))
    review_total = len(filtered_rows)
    testing_total = len(filtered_rows)

    review_prob = (review_feedback_total / review_total) if review_total else 0.0
    testing_prob = (testing_feedback_total / testing_total) if testing_total else 0.0

    header_keys = set(filtered_rows[0].keys())
    metadata = {
        "review_columns_used": [
            col for col in (REVIEW_COUNT_COLUMN, REVIEW_STATES_COLUMN) if col in header_keys
        ],
        "test_columns_used": [TEST_CONCLUSIONS_COLUMN] if TEST_CONCLUSIONS_COLUMN in header_keys else [],
        "created_missing": [str(missing_created)],
    }

    return (
        {
            "review_to_development": {
                "tickets_with_signal": review_total,
                "tickets_with_feedback": review_feedback_total,
                "probability": review_prob,
            },
            "testing_to_development": {
                "tickets_with_signal": testing_total,
                "tickets_with_feedback": testing_feedback_total,
                "probability": testing_prob,
            },
        },
        metadata,
    )


def write_output(metrics: dict[str, dict[str, float | int]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "tickets_with_signal", "tickets_with_feedback", "probability"])
        for metric, values in metrics.items():
            writer.writerow(
                [
                    metric,
                    values["tickets_with_signal"],
                    values["tickets_with_feedback"],
                    f"{values['probability']:.6f}",
                ]
            )


def write_enriched_csv(pr_rows: list[dict[str, str]], header: list[str], added_columns: list[str]) -> None:
    if not added_columns:
        return
    ENRICHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = header + [col for col in added_columns if col not in header]
    with ENRICHED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pr_rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute review/testing feedback probabilities.")
    parser.add_argument("--start", help="Start timestamp (inclusive) in ISO8601.")
    parser.add_argument("--end", help="End timestamp (exclusive) in ISO8601.")
    parser.add_argument(
        "--created-col",
        default="",
        help="Column name to use for created timestamp filtering.",
    )
    return parser.parse_args()


def select_created_col(header: list[str], created_col: str | None) -> str:
    if created_col:
        if created_col not in header:
            raise ValueError(f"Created column '{created_col}' not found in CSV header.")
        return created_col
    for candidate in CREATED_COL_CANDIDATES:
        if candidate in header:
            return candidate
    raise ValueError(
        "No created timestamp column found in CSV header. "
        f"Tried: {', '.join(CREATED_COL_CANDIDATES)}."
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    logging.info("Loading PR data from %s", PRS_CSV)
    pr_rows, header = load_pr_rows()
    logging.info("Loaded %d PR rows", len(pr_rows))
    logging.info("Loaded %d Jira rows", len(load_jira_rows()))
    ensure_required_signal_columns(header)
    enriched_rows, added_columns = enrich_pr_rows(pr_rows, header)
    write_enriched_csv(enriched_rows, header, added_columns)
    created_col = select_created_col(header, args.created_col.strip() or None)
    metrics, metadata = compute_feedback_probabilities(
        enriched_rows,
        created_col=created_col,
        start_str=args.start,
        end_str=args.end,
    )
    logging.info("Feedback time window: [%s, %s)", args.start or "MIN", args.end or "MAX")
    logging.info("Created timestamp column: %s", created_col)
    logging.info("Review columns used: %s", metadata["review_columns_used"])
    logging.info("Test columns used: %s", metadata["test_columns_used"])
    logging.info("Rows missing created timestamp: %s", metadata["created_missing"][0])
    for metric, values in metrics.items():
        logging.info(
            "%s: tickets_with_signal=%s tickets_with_feedback=%s probability=%.6f",
            metric,
            values["tickets_with_signal"],
            values["tickets_with_feedback"],
            values["probability"],
        )
    write_output(metrics)
    logging.info("Wrote feedback probabilities to %s", OUTPUT_CSV)
    if added_columns:
        logging.info("Wrote enriched feedback CSV to %s", ENRICHED_CSV)


if __name__ == "__main__":
    main()
