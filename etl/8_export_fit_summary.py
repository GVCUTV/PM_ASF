# v1
# file: 8_export_fit_summary.py

"""
Esporta un riepilogo compatto dei fit di distribuzione per ogni fase.

Legge i CSV prodotti da 7_fit_distributions.py (distribution_fit_stats_<stage>.csv)
seleziona il miglior fit e genera fit_summary.csv con parametri compatibili SciPy.
"""

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd

from path_config import PROJECT_ROOT


DEFAULT_STAGES = ["development", "review", "testing"]
STAGE_ALIASES = {
    "dev": "development",
    "development": "development",
    "review": "review",
    "test": "testing",
    "testing": "testing",
}
DIST_MAP = {
    "Lognormale": "lognorm",
    "Weibull": "weibull_min",
    "Esponenziale": "expon",
    "Normale": "norm",
}
PARAM_FIELDS = {
    "lognorm": ("s", "loc", "scale"),
    "weibull_min": ("c", "loc", "scale"),
    "expon": ("loc", "scale"),
    "norm": ("loc", "scale"),
}
DEFAULT_METRICS = ["MSE_KDE_PDF", "MAE_KDE_PDF", "MSE", "MAE"]


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


def parse_params(value):
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    text = str(value)
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return [float(m) for m in matches]


def normalize_stage(stage):
    key = stage.strip().lower()
    if key in STAGE_ALIASES:
        return STAGE_ALIASES[key]
    raise ValueError(f"Stage non riconosciuto: {stage}")


def choose_winner(stats_df, metric, plausible_only):
    if plausible_only and "Plausible" in stats_df.columns:
        plausible = stats_df[stats_df["Plausible"].astype(bool)]
        if plausible.empty:
            raise ValueError("Nessun fit plausibile disponibile.")
        stats_df = plausible
    if metric not in stats_df.columns:
        raise ValueError("Colonna metrica assente per la selezione del fit.")
    stats_df = stats_df.copy()
    stats_df[metric] = pd.to_numeric(stats_df[metric], errors="coerce")
    stats_df = stats_df.dropna(subset=[metric])
    if stats_df.empty:
        raise ValueError("Valori metrici non validi per la selezione del fit.")
    sort_cols = [metric]
    for extra in ("AIC", "BIC"):
        if extra in stats_df.columns:
            sort_cols.append(extra)
    return stats_df.sort_values(by=sort_cols).iloc[0]


def to_fit_summary_row(stage, winner_row, metric_name):
    params = parse_params(winner_row["Parametri"])
    dist_label = winner_row["Distribuzione"]
    dist = DIST_MAP.get(dist_label)
    if dist is None:
        raise ValueError(f"Distribuzione non supportata: {dist_label}")

    row = {
        "stage": stage,
        "dist": dist,
        "s": np.nan,
        "c": np.nan,
        "loc": np.nan,
        "scale": np.nan,
        "mean": winner_row.get("Mean"),
        "std": winner_row.get("Std"),
        "mse": winner_row.get(metric_name),
    }

    fields = PARAM_FIELDS.get(dist, ())
    for field_name, value in zip(fields, params):
        row[field_name] = value

    return row


def _resolve_stage_csv(base_dir, stage, overrides):
    if stage in overrides:
        return overrides[stage]
    return os.path.join(base_dir, f"distribution_fit_stats_{stage}.csv")


def parse_stage_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Formato override non valido. Usa stage=percorso.")
        stage_raw, path = item.split("=", 1)
        stage = normalize_stage(stage_raw)
        overrides[stage] = path
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Esporta un fit_summary.csv compatibile SciPy dalle statistiche di fit."  # noqa: E501
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.join(PROJECT_ROOT, "etl", "output", "csv"),
        help="Directory base dei CSV di fit (default: etl/output/csv).",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(PROJECT_ROOT, "etl", "output", "csv", "fit_summary.csv"),
        help="Percorso output per fit_summary.csv.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=DEFAULT_STAGES,
        help="Lista di stage da processare (dev/development, review, test/testing).",
    )
    parser.add_argument(
        "--stage-csv",
        action="append",
        default=[],
        help="Override path per stage: stage=percorso_csv. Ripetibile.",
    )
    parser.add_argument(
        "--plausible-only",
        action="store_true",
        help="Seleziona solo fit plausibili quando disponibile.",
    )
    parser.add_argument(
        "--log-path",
        default=os.path.join(PROJECT_ROOT, "etl", "output", "logs", "export_fit_summary.log"),
        help="Percorso del log file.",
    )

    args = parser.parse_args()
    _setup_logging(args.log_path)

    stages = [normalize_stage(stage) for stage in args.stages]
    overrides = parse_stage_overrides(args.stage_csv)
    summary_rows = []

    for stage in stages:
        stats_path = _resolve_stage_csv(args.base_dir, stage, overrides)
        if not os.path.exists(stats_path):
            raise SystemExit(f"CSV non trovato per {stage}: {stats_path}")
        try:
            stats_df = pd.read_csv(stats_path)
        except Exception as exc:
            logging.error("Errore nel caricamento CSV %s: %s", stats_path, exc)
            raise SystemExit(1) from exc

        required_cols = {"Distribuzione", "Parametri"}
        missing = required_cols - set(stats_df.columns)
        if missing:
            raise SystemExit(f"Colonne mancanti in {stats_path}: {sorted(missing)}")

        metric = next((m for m in DEFAULT_METRICS if m in stats_df.columns), None)
        if metric is None:
            raise SystemExit(
                f"Nessuna colonna metrica trovata in {stats_path}."
            )

        try:
            winner = choose_winner(stats_df, metric, args.plausible_only)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        summary_rows.append(to_fit_summary_row(stage, winner, metric))
        logging.info("Stage %s: selezionato %s", stage, winner["Distribuzione"])

    if not summary_rows:
        raise SystemExit("Nessun fit valido: output non generato.")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_csv, index=False)
    logging.info("Fit summary salvato in %s", args.out_csv)


if __name__ == "__main__":
    main()
