# v1
# file: etl/4_feedback_probabilities_etl.py
"""
Extract ticket-level feedback probabilities for:
- Review -> Development (changes requested during review)
- Testing -> Development (CI/check failures during testing)
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

PRS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "github_prs_raw.csv"
OUTPUT_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "feedback_probabilities.csv"
LOG_PATH = Path(PROJECT_ROOT) / "etl" / "output" / "logs" / "feedback_probabilities.log"

JIRA_KEY_REGEX = re.compile(r"BOOKKEEPER-\d+", re.IGNORECASE)
REVIEW_CHANGE_STATES = {"CHANGES_REQUESTED"}
CHECK_FAILURES = {"failure", "cancelled", "timed_out", "action_required", "startup_failure"}
STATUS_FAILURES = {"failure", "error"}


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


def parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def extract_jira_keys(*values: str | None) -> set[str]:
    keys: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in JIRA_KEY_REGEX.findall(str(value)):
            keys.add(match.upper())
    return keys


def parse_requested_changes(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def review_feedback_from_row(row: dict[str, str]) -> tuple[bool, bool]:
    requested_changes = parse_requested_changes(row.get("requested_changes_count"))
    review_states = [state.upper() for state in parse_json_list(row.get("pull_request_review_states"))]
    has_signal = bool(row.get("requested_changes_count") or review_states)
    has_feedback = requested_changes > 0 or any(state in REVIEW_CHANGE_STATES for state in review_states)
    return has_signal, has_feedback


def testing_feedback_from_row(row: dict[str, str]) -> tuple[bool, bool]:
    check_runs = [state.lower() for state in parse_json_list(row.get("check_runs_conclusions"))]
    statuses = [state.lower() for state in parse_json_list(row.get("combined_status_states"))]
    has_signal = bool(check_runs or statuses)
    has_feedback = any(state in CHECK_FAILURES for state in check_runs) or any(
        state in STATUS_FAILURES for state in statuses
    )
    return has_signal, has_feedback


def load_pr_rows() -> list[dict[str, str]]:
    if not PRS_CSV.exists():
        raise FileNotFoundError(f"Missing PR CSV: {PRS_CSV}")
    with PRS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def compute_feedback_probabilities(pr_rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    ticket_flags: dict[str, TicketFeedback] = {}
    for row in pr_rows:
        jira_keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not jira_keys:
            continue
        review_signal, review_feedback = review_feedback_from_row(row)
        testing_signal, testing_feedback = testing_feedback_from_row(row)
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

    return {
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
    }


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


def main() -> None:
    setup_logging()
    logging.info("Loading PR data from %s", PRS_CSV)
    pr_rows = load_pr_rows()
    logging.info("Loaded %d PR rows", len(pr_rows))
    metrics = compute_feedback_probabilities(pr_rows)
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


if __name__ == "__main__":
    main()
