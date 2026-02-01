# v1
# file: etl/4_feedback_probabilities_etl.py
"""
Extract ticket-level feedback probabilities for:
- Review -> Development (changes requested during review)
- Testing -> Development (CI/check failures during testing)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

PRS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "github_prs_raw.csv"
JIRA_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_issues_raw.csv"
OUTPUT_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "feedback_probabilities.csv"
ENRICHED_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "feedback_enriched.csv"
LOG_PATH = Path(PROJECT_ROOT) / "etl" / "output" / "logs" / "feedback_probabilities.log"

JIRA_KEY_REGEX = re.compile(r"BOOKKEEPER-\d+", re.IGNORECASE)

CREATED_COL_CANDIDATES = ("created_at", "created", "fields.created")

REVIEW_NUMERIC_COLUMNS = (
    "review_rounds",
    "pr_review_rounds",
    "requested_changes_count",
    "reviews_count",
)
REVIEW_FLAG_COLUMNS = ("review_rework_flag", "reopened_flag", "requested_changes_flag")
REVIEW_LIST_COLUMNS = (
    "pull_request_review_states",
    "review_states",
    "pr_review_states",
    "review_decisions",
    "requested_changes_states",
)

TEST_FLAG_COLUMNS = ("ci_failed_then_fix", "ci_failed", "build_failed", "qa_failed_flag")
TEST_LIST_COLUMNS = (
    "check_runs_conclusions",
    "ci_status_history",
    "combined_statuses",
    "workflow_conclusions",
    "build_state_history",
    "statuses",
    "ci_conclusion",
    "check_suite_conclusion",
    "build_conclusion",
)

TRUTHY_VALUES = {"true", "1", "yes", "y", "t"}
FAILURE_TOKENS = {
    "fail",
    "failure",
    "failed",
    "error",
    "timed_out",
    "timeout",
    "cancelled",
    "canceled",
    "aborted",
    "broken",
}
SUCCESS_TOKENS = {"success", "succeeded", "passed", "ok", "green", "completed_success"}


@dataclass
class TicketFeedback:
    has_review_signal: bool = False
    has_review_feedback: bool = False
    has_testing_signal: bool = False
    has_testing_feedback: bool = False


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


def extract_jira_keys(*values: str | None) -> set[str]:
    keys: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in JIRA_KEY_REGEX.findall(str(value)):
            keys.add(match.upper())
    return keys


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES


def _truthy_fraction(values: list[object]) -> float:
    if not values:
        return 0.0
    truthy = sum(1 for value in values if _is_truthy(value))
    return truthy / len(values)


def _has_fail_then_success(tokens: list[str]) -> bool:
    lowered = [token.strip().lower() for token in tokens if token and str(token).strip()]
    failure_indices = [
        idx for idx, token in enumerate(lowered) if any(failure in token for failure in FAILURE_TOKENS)
    ]
    if not failure_indices:
        return False
    first_failure = min(failure_indices)
    for token in lowered[first_failure + 1 :]:
        if any(success in token for success in SUCCESS_TOKENS):
            return True
    return False


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


def review_feedback_from_row(
    row: dict[str, str],
    numeric_columns: list[str],
    flag_columns: list[str],
    list_columns: list[str],
) -> tuple[bool, bool]:
    if numeric_columns:
        values = [_parse_float(row.get(col)) for col in numeric_columns]
        numeric_values = [value for value in values if value is not None]
        has_signal = bool(numeric_values)
        has_feedback = bool(numeric_values and max(numeric_values) > 1)
        return has_signal, has_feedback
    if flag_columns:
        flags = [row.get(col) for col in flag_columns]
        has_signal = any(value not in (None, "") for value in flags)
        has_feedback = any(_is_truthy(value) for value in flags)
        return has_signal, has_feedback
    list_values = [_to_listish(row.get(col)) for col in list_columns]
    has_signal = any(tokens for tokens in list_values)
    has_feedback = any(
        "changes_requested" in token.lower()
        for tokens in list_values
        for token in tokens
        if token
    )
    return has_signal, has_feedback


def testing_feedback_from_row(
    row: dict[str, str],
    flag_columns: list[str],
    list_columns: list[str],
) -> tuple[bool, bool]:
    if flag_columns:
        flags = [row.get(col) for col in flag_columns]
        has_signal = any(value not in (None, "") for value in flags)
        has_feedback = any(_is_truthy(value) for value in flags)
        return has_signal, has_feedback
    list_values = [_to_listish(row.get(col)) for col in list_columns]
    has_signal = any(tokens for tokens in list_values)
    has_feedback = any(_has_fail_then_success(tokens) for tokens in list_values)
    return has_signal, has_feedback


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
    review_columns = set(REVIEW_NUMERIC_COLUMNS + REVIEW_FLAG_COLUMNS + REVIEW_LIST_COLUMNS)
    test_columns = set(TEST_FLAG_COLUMNS + TEST_LIST_COLUMNS + ("combined_status_states",))
    review_available = review_columns.intersection(header)
    test_available = test_columns.intersection(header)
    if not review_available:
        raise ValueError(
            "Missing review feedback signals in raw data. "
            f"Expected one of: {sorted(review_columns)}. "
            "Please export the required review signal columns."
        )
    if not test_available:
        raise ValueError(
            "Missing testing/CI feedback signals in raw data. "
            f"Expected one of: {sorted(test_columns)}. "
            "Please export the required CI/status signal columns."
        )


def derive_review_rounds(row: dict[str, str]) -> float | None:
    for col in ("review_rounds", "pr_review_rounds", "requested_changes_count", "reviews_count"):
        value = _parse_float(row.get(col))
        if value is not None:
            return value
    for col in REVIEW_LIST_COLUMNS:
        tokens = _to_listish(row.get(col))
        if tokens:
            has_changes = any("changes_requested" in token.lower() for token in tokens)
            return 2.0 if has_changes else 1.0
    return None


def derive_review_rework_flag(row: dict[str, str], review_rounds: float | None) -> bool:
    if review_rounds is not None and review_rounds > 1:
        return True
    for col in REVIEW_LIST_COLUMNS:
        tokens = _to_listish(row.get(col))
        if any("changes_requested" in token.lower() for token in tokens):
            return True
    return False


def derive_ci_failed_then_fix(row: dict[str, str]) -> bool:
    flag_values = [row.get(col) for col in TEST_FLAG_COLUMNS if col in row]
    if any(_is_truthy(value) for value in flag_values):
        return True
    list_columns = [col for col in TEST_LIST_COLUMNS if col in row]
    if "combined_statuses" not in row and "combined_status_states" in row:
        list_columns.append("combined_status_states")
    for col in list_columns:
        if _has_fail_then_success(_to_listish(row.get(col))):
            return True
    return False


def enrich_pr_rows(pr_rows: list[dict[str, str]], header: list[str]) -> tuple[list[dict[str, str]], list[str]]:
    enriched_rows: list[dict[str, str]] = []
    added_columns: list[str] = []
    needs_review_rounds = "review_rounds" not in header
    needs_review_rework = "review_rework_flag" not in header
    needs_ci_failed = "ci_failed_then_fix" not in header
    needs_combined_statuses = "combined_statuses" not in header and "combined_status_states" in header

    if needs_review_rounds:
        added_columns.append("review_rounds")
    if needs_review_rework:
        added_columns.append("review_rework_flag")
    if needs_ci_failed:
        added_columns.append("ci_failed_then_fix")
    if needs_combined_statuses:
        added_columns.append("combined_statuses")

    for row in pr_rows:
        updated = dict(row)
        review_rounds = derive_review_rounds(updated) if needs_review_rounds else _parse_float(updated.get("review_rounds"))
        if needs_review_rounds:
            updated["review_rounds"] = "" if review_rounds is None else str(review_rounds)
        if needs_review_rework:
            review_rework = derive_review_rework_flag(updated, review_rounds)
            updated["review_rework_flag"] = "true" if review_rework else "false"
        if needs_ci_failed:
            ci_failed = derive_ci_failed_then_fix(updated)
            updated["ci_failed_then_fix"] = "true" if ci_failed else "false"
        if needs_combined_statuses:
            updated["combined_statuses"] = updated.get("combined_status_states", "")
        enriched_rows.append(updated)
    return enriched_rows, added_columns


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

    numeric_columns = [
        col
        for col in REVIEW_NUMERIC_COLUMNS
        if any(_parse_float(row.get(col)) is not None for row in filtered_rows)
    ]
    flag_columns = [
        col
        for col in REVIEW_FLAG_COLUMNS
        if any(row.get(col) not in (None, "") for row in filtered_rows)
    ]
    list_columns = [
        col
        for col in REVIEW_LIST_COLUMNS
        if any(_to_listish(row.get(col)) for row in filtered_rows)
    ]
    review_numeric = numeric_columns
    review_flags = [] if numeric_columns else flag_columns
    review_lists = [] if (numeric_columns or flag_columns) else list_columns
    if not (review_numeric or review_flags or review_lists):
        raise ValueError("No review feedback signals available in the time window slice.")

    test_flags = [
        col
        for col in TEST_FLAG_COLUMNS
        if any(row.get(col) not in (None, "") for row in filtered_rows)
    ]
    test_lists = [
        col
        for col in TEST_LIST_COLUMNS
        if any(_to_listish(row.get(col)) for row in filtered_rows)
    ]
    if "combined_statuses" not in test_lists and any(
        _to_listish(row.get("combined_status_states")) for row in filtered_rows
    ):
        test_lists.append("combined_statuses")
        for row in filtered_rows:
            if "combined_statuses" not in row and "combined_status_states" in row:
                row["combined_statuses"] = row.get("combined_status_states", "")
    test_flags = test_flags
    test_lists = [] if test_flags else test_lists
    if not (test_flags or test_lists):
        raise ValueError("No testing/CI feedback signals available in the time window slice.")

    ticket_flags: dict[str, TicketFeedback] = {}
    for row in filtered_rows:
        jira_keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not jira_keys:
            continue
        review_signal, review_feedback = review_feedback_from_row(row, review_numeric, review_flags, review_lists)
        testing_signal, testing_feedback = testing_feedback_from_row(row, test_flags, test_lists)
        for key in jira_keys:
            feedback = ticket_flags.setdefault(key, TicketFeedback())
            feedback.has_review_signal = feedback.has_review_signal or review_signal
            feedback.has_review_feedback = feedback.has_review_feedback or review_feedback
            feedback.has_testing_signal = feedback.has_testing_signal or testing_signal
            feedback.has_testing_feedback = feedback.has_testing_feedback or testing_feedback

    review_total = sum(1 for feedback in ticket_flags.values() if feedback.has_review_signal)
    review_feedback_total = sum(1 for feedback in ticket_flags.values() if feedback.has_review_feedback)
    testing_total = sum(1 for feedback in ticket_flags.values() if feedback.has_testing_signal)
    testing_feedback_total = sum(1 for feedback in ticket_flags.values() if feedback.has_testing_feedback)

    review_prob = (review_feedback_total / review_total) if review_total else 0.0
    testing_prob = (testing_feedback_total / testing_total) if testing_total else 0.0

    metadata = {
        "review_numeric_columns": review_numeric,
        "review_flag_columns": review_flags,
        "review_list_columns": review_lists,
        "test_flag_columns": test_flags,
        "test_list_columns": test_lists,
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
    logging.info("Review numeric columns used: %s", metadata["review_numeric_columns"])
    logging.info("Review flag columns used: %s", metadata["review_flag_columns"])
    logging.info("Review list columns used: %s", metadata["review_list_columns"])
    logging.info("Test flag columns used: %s", metadata["test_flag_columns"])
    logging.info("Test list columns used: %s", metadata["test_list_columns"])
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
