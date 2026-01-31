# v1
# file: 6_diagnose_and_plot_tickets.py

"""
Diagnostica per-ticket e distribuzione dei tempi di risoluzione.

- Carica il dataset mergeato.
- Calcola il tempo di risoluzione in ore (se possibile).
- Stampa per ogni ticket i campi chiave e segnala anomalie.
- Produce un istogramma dei tempi di risoluzione tra 0 e 10.000 ore.
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
OUT_PNG = PROJECT_ROOT + "/etl/output/png/distribuzione_resolution_times_0_10000.png"
LOG_PATH = PROJECT_ROOT + "/etl/output/logs/diagnose_tickets.log"

# === Logging ===
os.makedirs(PROJECT_ROOT + "/etl/output/logs", exist_ok=True)
os.makedirs(PROJECT_ROOT + "/etl/output/png", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)


def _resolve_col(df, primary, fallback):
    if primary in df.columns:
        return primary
    if fallback in df.columns:
        return fallback
    return None


def _to_datetime(df, col):
    if col and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
        logging.info("Convertita colonna a datetime: %s", col)


def _compute_resolution_hours(df, created_col, resolved_col):
    if created_col and resolved_col and created_col in df.columns and resolved_col in df.columns:
        return (df[resolved_col] - df[created_col]).dt.total_seconds().div(3600.0)
    if "resolution_time_days" in df.columns:
        return pd.to_numeric(df["resolution_time_days"], errors="coerce").mul(24.0)
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def _flags_for_row(row, key_col, created_col, resolved_col, status_col, resolution_hours):
    flags = []
    key_val = row.get(key_col) if key_col else None
    created_val = row.get(created_col) if created_col else None
    resolved_val = row.get(resolved_col) if resolved_col else None
    status_val = row.get(status_col) if status_col else None

    if key_col is None or pd.isna(key_val):
        flags.append("missing_key")
    if created_col is None or pd.isna(created_val):
        flags.append("missing_created")
    if resolved_col is None or pd.isna(resolved_val):
        if status_val in {"Closed", "Resolved"}:
            flags.append("closed_without_resolution")
    if created_col and resolved_col and pd.notna(created_val) and pd.notna(resolved_val):
        if resolved_val < created_val:
            flags.append("resolution_before_creation")
    if pd.notna(resolution_hours) and resolution_hours < 0:
        flags.append("negative_resolution_time")

    return flags


def _print_ticket_row(row, key_col, created_col, resolved_col, status_col, type_col, resolution_hours, flags):
    key_val = row.get(key_col) if key_col else None
    created_val = row.get(created_col) if created_col else None
    resolved_val = row.get(resolved_col) if resolved_col else None
    status_val = row.get(status_col) if status_col else None
    type_val = row.get(type_col) if type_col else None

    flags_str = ",".join(flags) if flags else "ok"
    print(
        f"key={key_val} | type={type_val} | status={status_val} | "
        f"created={created_val} | resolved={resolved_val} | "
        f"resolution_hours={resolution_hours} | flags={flags_str}"
    )


if __name__ == "__main__":
    try:
        df = pd.read_csv(IN_CSV, low_memory=False)
    except Exception as exc:
        logging.error("Errore nel caricamento CSV: %s", exc)
        raise SystemExit(1) from exc

    key_col = _resolve_col(df, "key", "jira_key")
    created_col = _resolve_col(df, "fields.created", "created")
    resolved_col = _resolve_col(df, "fields.resolutiondate", "resolved")
    status_col = _resolve_col(df, "fields.status.name", "status")
    type_col = _resolve_col(df, "fields.issuetype.name", "issuetype")

    _to_datetime(df, created_col)
    _to_datetime(df, resolved_col)

    resolution_hours = _compute_resolution_hours(df, created_col, resolved_col)
    df["resolution_time_hours"] = resolution_hours

    logging.info("Totale righe: %d", len(df))

    anomaly_count = 0
    for _, row in df.iterrows():
        flags = _flags_for_row(row, key_col, created_col, resolved_col, status_col, row.get("resolution_time_hours"))
        if flags:
            anomaly_count += 1
            logging.warning("Ticket con anomalie (%s): %s", ",".join(flags), row.get(key_col))
        _print_ticket_row(
            row,
            key_col,
            created_col,
            resolved_col,
            status_col,
            type_col,
            row.get("resolution_time_hours"),
            flags,
        )

    logging.info("Ticket con anomalie rilevate: %d", anomaly_count)

    valid_hours = pd.to_numeric(df["resolution_time_hours"], errors="coerce").dropna()
    valid_hours = valid_hours[(valid_hours >= 0) & (valid_hours <= 10000)]

    if valid_hours.empty:
        logging.warning("Nessun dato valido per histogram: salto grafico.")
    else:
        bins = int(np.clip(np.sqrt(len(valid_hours)), 10, 100))
        plt.figure(figsize=(10, 5))
        plt.hist(valid_hours, bins=bins, color="#4c72b0", edgecolor="white")
        plt.title("Distribuzione tempi di risoluzione (0-10000 ore)")
        plt.xlabel("Ore di risoluzione")
        plt.ylabel("Numero di ticket")
        plt.tight_layout()
        plt.savefig(OUT_PNG)
        logging.info("Grafico salvato in %s", OUT_PNG)
