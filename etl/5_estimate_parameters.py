# v1
# file: 5_estimate_parameters.py

"""
Stima parametri globali e statistiche di fase dal dataset mergeato.

Output principali:
- parameter_estimates.csv (arrivi, throughput, tempi di risoluzione)
- phase_durations_wide.csv (durate per fase in formato wide)
- phase_summary_stats.csv (statistiche per fase)
- backlog_over_time.png (backlog giornaliero)
"""

import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_config import PROJECT_ROOT

matplotlib.use("Agg")

# === Path I/O ===
IN_CSV = PROJECT_ROOT + "/etl/output/csv/tickets_prs_merged.csv"
OUT_PARAMS_CSV = PROJECT_ROOT + "/etl/output/csv/parameter_estimates.csv"
OUT_PHASE_WIDE_CSV = PROJECT_ROOT + "/etl/output/csv/phase_durations_wide.csv"
OUT_PHASE_SUMMARY_CSV = PROJECT_ROOT + "/etl/output/csv/phase_summary_stats.csv"
OUT_BACKLOG_PNG = PROJECT_ROOT + "/etl/output/png/backlog_over_time.png"

# === Logging ===
os.makedirs(PROJECT_ROOT + "/etl/output/logs", exist_ok=True)
os.makedirs(PROJECT_ROOT + "/etl/output/csv", exist_ok=True)
os.makedirs(PROJECT_ROOT + "/etl/output/png", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT + "/etl/output/logs/estimate_parameters.log"),
        logging.StreamHandler()
    ]
)


def _to_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
        logging.info("Convertita colonna a datetime: %s", col)
    else:
        logging.warning("Colonna assente: %s", col)


def _resolve_col(df, primary, fallback):
    if primary in df.columns:
        return primary
    if fallback in df.columns:
        return fallback
    return None


def summarize_phase(series, phase_name):
    values = pd.to_numeric(series, errors="coerce")
    total = len(values)
    valid = values.dropna()
    if total == 0:
        missing_share = np.nan
    else:
        missing_share = 1.0 - (len(valid) / total)

    if len(valid) == 0:
        return {
            "phase": phase_name,
            "count": 0,
            "missing_share": missing_share,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }

    return {
        "phase": phase_name,
        "count": len(valid),
        "missing_share": missing_share,
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "p25": float(valid.quantile(0.25)),
        "p75": float(valid.quantile(0.75)),
        "max": float(valid.max()),
    }


def estimate_arrival_rate(df, created_col):
    if created_col is None:
        logging.warning("Colonna creazione assente: impossibile stimare arrivi.")
        return np.nan, pd.Series(dtype=float)
    created = df[created_col].dropna().sort_values()
    interarrival = created.diff().dt.total_seconds().div(86400.0)
    mean_interarrival = interarrival.dropna().mean()
    if mean_interarrival and mean_interarrival > 0:
        return 1.0 / mean_interarrival, interarrival
    return np.nan, interarrival


def estimate_throughput_monthly(df, resolved_col):
    if resolved_col is None:
        logging.warning("Colonna risoluzione assente: impossibile stimare throughput.")
        return np.nan
    resolved = df[resolved_col].dropna()
    if resolved.empty:
        return np.nan
    counts = resolved.dt.to_period("M").value_counts().sort_index()
    return float(counts.mean()) if not counts.empty else np.nan


def compute_backlog(df, created_col, resolved_col):
    if created_col is None:
        logging.warning("Colonna creazione assente: backlog non calcolabile.")
        return None

    created = df[created_col].dropna().dt.floor("D")
    if created.empty:
        logging.warning("Nessuna data di creazione valida: backlog vuoto.")
        return None

    resolved = pd.Series([], dtype="datetime64[ns]")
    if resolved_col and resolved_col in df.columns:
        resolved = df[resolved_col].dropna().dt.floor("D")

    created_counts = created.value_counts().sort_index()
    resolved_counts = resolved.value_counts().sort_index()

    start_date = created_counts.index.min()
    end_candidates = [created_counts.index.max()]
    if not resolved_counts.empty:
        end_candidates.append(resolved_counts.index.max())
    end_date = max(end_candidates)

    date_index = pd.date_range(start=start_date, end=end_date, freq="D")
    created_daily = created_counts.reindex(date_index, fill_value=0)
    resolved_daily = resolved_counts.reindex(date_index, fill_value=0)
    backlog = created_daily.cumsum() - resolved_daily.cumsum()

    return backlog


if __name__ == "__main__":
    try:
        df = pd.read_csv(IN_CSV, low_memory=False)
    except Exception as exc:
        logging.error("Errore nel caricamento CSV: %s", exc)
        raise SystemExit(1) from exc

    created_col = _resolve_col(df, "fields.created", "created")
    resolved_col = _resolve_col(df, "fields.resolutiondate", "resolved")

    if created_col:
        _to_datetime(df, created_col)
    if resolved_col:
        _to_datetime(df, resolved_col)

    if "resolution_time_days" not in df.columns and created_col and resolved_col:
        df["resolution_time_days"] = (
            df[resolved_col] - df[created_col]
        ).dt.total_seconds().div(86400.0)
        logging.info("Calcolata colonna resolution_time_days.")

    arrival_rate_per_day, interarrival = estimate_arrival_rate(df, created_col)
    mean_resolution_time_days = pd.to_numeric(
        df.get("resolution_time_days"), errors="coerce"
    ).dropna().mean()
    median_resolution_time_days = pd.to_numeric(
        df.get("resolution_time_days"), errors="coerce"
    ).dropna().median()
    throughput_monthly_mean = estimate_throughput_monthly(df, resolved_col)

    params = pd.DataFrame([
        {
            "arrival_rate_per_day": arrival_rate_per_day,
            "mean_resolution_time_days": mean_resolution_time_days,
            "median_resolution_time_days": median_resolution_time_days,
            "throughput_monthly_mean": throughput_monthly_mean,
            "mean_interarrival_days": interarrival.dropna().mean() if not interarrival.empty else np.nan,
        }
    ])
    params.to_csv(OUT_PARAMS_CSV, index=False)
    logging.info("Parametri globali salvati in %s", OUT_PARAMS_CSV)

    phase_cols = [
        ("dev_duration_days", "development"),
        ("review_duration_days", "review"),
        ("test_duration_days", "testing"),
    ]

    phase_rows = []
    for col, label in phase_cols:
        if col in df.columns:
            phase_rows.append(summarize_phase(df[col], label))
        else:
            logging.warning("Colonna fase mancante: %s", col)
            phase_rows.append(summarize_phase(pd.Series([], dtype=float), label))

    phase_summary = pd.DataFrame(phase_rows)
    phase_summary.to_csv(OUT_PHASE_SUMMARY_CSV, index=False)
    logging.info("Statistiche di fase salvate in %s", OUT_PHASE_SUMMARY_CSV)

    phase_wide_cols = [col for col, _ in phase_cols if col in df.columns]
    if "key" in df.columns:
        phase_wide_cols = ["key"] + phase_wide_cols
    if phase_wide_cols:
        df[phase_wide_cols].to_csv(OUT_PHASE_WIDE_CSV, index=False)
        logging.info("Durate di fase salvate in %s", OUT_PHASE_WIDE_CSV)
    else:
        logging.warning("Nessuna colonna di durata trovata: salto export phase_durations_wide.csv")

    backlog = compute_backlog(df, created_col, resolved_col)
    if backlog is not None:
        plt.figure(figsize=(10, 4))
        backlog.plot()
        plt.title("Backlog giornaliero (ticket aperti)")
        plt.xlabel("Data")
        plt.ylabel("Ticket aperti")
        plt.tight_layout()
        plt.savefig(OUT_BACKLOG_PNG)
        logging.info("Backlog salvato in %s", OUT_BACKLOG_PNG)
    else:
        logging.warning("Backlog non calcolato: salto grafico.")
