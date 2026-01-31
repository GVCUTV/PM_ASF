# v1
# file: 9_enrich_feedback_cols.py

"""
Arricchisce tickets_prs_merged.csv con segnali di feedback (review/CI) e ruoli inferiti.

Deriva:
- review_rounds: numero di review round (da contatori o stati review).
- review_rework_flag: True se risultano richieste di cambiamento.
- ci_failed_then_fix: True se CI mostra un fallimento seguito da successo.
- dev_user: autore/assegnatario stimato.
- tester: reviewer/QA stimato.
"""

import argparse
import ast
import json
import logging
import os

import numpy as np
import pandas as pd

from path_config import PROJECT_ROOT


DEFAULT_IN_CSV = os.path.join(PROJECT_ROOT, "etl", "output", "csv", "tickets_prs_merged.csv")
DEFAULT_OUT_CSV = DEFAULT_IN_CSV
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "etl", "output", "logs", "enrich_feedback.log")

FAIL_TOKENS = {
    "failure",
    "failed",
    "cancelled",
    "canceled",
    "timed_out",
    "action_required",
    "error",
    "startup_failure",
}
SUCCESS_TOKENS = {"success", "passed", "neutral"}


def _setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


def _to_listish(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [v for v in value if not pd.isna(v)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [item.strip() for item in text.split(",") if item.strip()]
    return [value]


def _has_fail_then_success(states):
    seen_fail = False
    for raw in states:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token in FAIL_TOKENS:
            seen_fail = True
            continue
        if seen_fail and token in SUCCESS_TOKENS:
            return True
    return False


def _list_len(series):
    return series.apply(lambda v: len(_to_listish(v)))


def _first_nonnull(row, candidates):
    for col in candidates:
        if col in row:
            value = row[col]
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                return text
    return pd.NA


def _first_from_list(value):
    items = [str(v).strip() for v in _to_listish(value) if str(v).strip()]
    return items[0] if items else pd.NA


def _infer_tester(row, fallback_cols):
    if "requested_reviewers" in row:
        reviewer = _first_from_list(row.get("requested_reviewers"))
        if pd.notna(reviewer):
            return reviewer
    return _first_nonnull(row, fallback_cols)


def enrich(df):
    review_rounds = pd.Series(np.nan, index=df.index)
    if "reviews_count" in df.columns:
        review_rounds = pd.to_numeric(df["reviews_count"], errors="coerce")

    if "pull_request_review_states" in df.columns:
        list_lengths = _list_len(df["pull_request_review_states"])
        review_rounds = review_rounds.combine_first(list_lengths)

    df["review_rounds"] = review_rounds

    rework_flag = pd.Series(False, index=df.index)
    if "requested_changes_count" in df.columns:
        requested_changes = pd.to_numeric(df["requested_changes_count"], errors="coerce")
        rework_flag = rework_flag | (requested_changes.fillna(0) > 0)

    if "pull_request_review_states" in df.columns:
        rework_flag = rework_flag | df["pull_request_review_states"].apply(
            lambda v: "CHANGES_REQUESTED" in [
                str(s).upper() for s in _to_listish(v)
            ]
        )

    df["review_rework_flag"] = rework_flag

    status_sources = []
    if "check_runs_conclusions" in df.columns:
        status_sources.append(df["check_runs_conclusions"])
    if "combined_status_states" in df.columns:
        status_sources.append(df["combined_status_states"])

    if status_sources:
        combined = status_sources[0].apply(_to_listish)
        for extra in status_sources[1:]:
            combined = combined + extra.apply(_to_listish)
        df["ci_failed_then_fix"] = combined.apply(_has_fail_then_success)
    else:
        logging.warning("Nessuna colonna CI trovata per derivare ci_failed_then_fix.")

    dev_candidates = [
        "user.login",
        "assignee.login",
        "fields.assignee.displayName",
        "fields.assignee.name",
        "fields.assignee.key",
        "fields.creator.displayName",
        "fields.creator.name",
    ]
    df["dev_user"] = df.apply(lambda row: _first_nonnull(row, dev_candidates), axis=1)

    tester_candidates = [
        "requested_reviewers",
        "reviewers",
        "fields.reporter.displayName",
        "fields.reporter.name",
        "fields.reporter.key",
    ]
    df["tester"] = df.apply(lambda row: _infer_tester(row, tester_candidates), axis=1)

    return df


def _log_coverage(df, columns):
    for col in columns:
        if col in df.columns:
            count = df[col].notna().sum()
            logging.info("Coverage %s: %d/%d", col, count, len(df))
        else:
            logging.warning("Colonna %s non presente nel dataset finale.", col)


def main():
    parser = argparse.ArgumentParser(
        description="Arricchisce il dataset merge con segnali review/CI e ruoli inferiti."
    )
    parser.add_argument(
        "--in-csv",
        default=DEFAULT_IN_CSV,
        help="Percorso input (default: etl/output/csv/tickets_prs_merged.csv)",
    )
    parser.add_argument(
        "--out-csv",
        default=DEFAULT_OUT_CSV,
        help="Percorso output CSV (default: sovrascrive input).",
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG,
        help="Percorso file di log.",
    )

    args = parser.parse_args()
    _setup_logging(args.log_path)

    logging.info("Input CSV: %s", args.in_csv)
    logging.info("Output CSV: %s", args.out_csv)

    try:
        df = pd.read_csv(args.in_csv)
    except Exception as exc:
        logging.error("Errore nel caricamento CSV %s: %s", args.in_csv, exc)
        raise SystemExit(1) from exc

    df = enrich(df)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logging.info("CSV arricchito salvato: %s", args.out_csv)

    _log_coverage(
        df,
        ["review_rounds", "review_rework_flag", "ci_failed_then_fix", "dev_user", "tester"],
    )


if __name__ == "__main__":
    main()
