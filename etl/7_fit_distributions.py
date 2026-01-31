# v1
# file: 7_fit_distributions.py

"""
Fit delle distribuzioni sui tempi di fase e sulla risoluzione complessiva.

Output principali:
- distribution_fit_stats.csv (fit su resolution_time_days)
- distribution_fit_stats_<stage>.csv (development/review/testing)
- fit_summary.csv (migliori fit per stage)
- confronto_fit_<stage>.png (grafici PDF vs KDE)
"""

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from path_config import PROJECT_ROOT

matplotlib.use("Agg")

# === Path I/O ===
IN_CSV = PROJECT_ROOT + "/etl/output/csv/tickets_prs_merged.csv"
OUT_CSV_DIR = PROJECT_ROOT + "/etl/output/csv"
OUT_PNG_DIR = PROJECT_ROOT + "/etl/output/png"
OUT_LOG = PROJECT_ROOT + "/etl/output/logs/fit_distributions.log"

# === Config ===
MIN_SAMPLES = 10
MAX_DAYS = 3650  # 10 anni
GRID_POINTS = 1000

DISTRIBUTIONS = [
    ("Lognormale", stats.lognorm),
    ("Weibull", stats.weibull_min),
    ("Esponenziale", stats.expon),
    ("Normale", stats.norm),
]

# === Logging ===
os.makedirs(PROJECT_ROOT + "/etl/output/logs", exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_PNG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUT_LOG),
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
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
        logging.info("Convertita colonna a datetime: %s", col)
    else:
        logging.warning("Colonna assente: %s", col)


def _clean_series(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values[(values >= 0) & (values <= MAX_DAYS)]
    return values


def _safe_kde(values):
    if len(values) < 2:
        return None
    if values.nunique() < 2:
        return None
    try:
        return stats.gaussian_kde(values)
    except Exception as exc:
        logging.warning("KDE fallita: %s", exc)
        return None


def _mean_std_from_params(dist, params):
    try:
        mean = dist.mean(*params)
        std = dist.std(*params)
    except Exception:
        return np.nan, np.nan
    return float(mean), float(std)


def _plausible(mean, std):
    if not np.isfinite(mean) or not np.isfinite(std):
        return False
    if mean < 0 or std <= 0:
        return False
    if mean > MAX_DAYS:
        return False
    return True


def _ks_aic_bic(dist, params, values):
    ks_pvalue = np.nan
    aic = np.nan
    bic = np.nan

    try:
        ks_stat = stats.kstest(values, dist.cdf, args=params)
        ks_pvalue = float(ks_stat.pvalue)
    except Exception as exc:
        logging.warning("KS test fallito per %s: %s", dist.name, exc)

    try:
        logpdf = dist.logpdf(values, *params)
        if np.isfinite(logpdf).all():
            ll = float(np.sum(logpdf))
            k = len(params)
            n = len(values)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
    except Exception as exc:
        logging.warning("AIC/BIC falliti per %s: %s", dist.name, exc)

    return ks_pvalue, aic, bic


def _fit_distribution_set(values, label, out_csv, out_png):
    cleaned = _clean_series(values)
    if len(cleaned) < MIN_SAMPLES:
        logging.warning(
            "Campioni insufficienti per %s: %s", label, len(cleaned)
        )
        return None

    kde = _safe_kde(cleaned)
    if kde is None:
        logging.warning("KDE non disponibile per %s", label)
        return None

    x_min = 0.0
    x_max = max(cleaned.max(), 1e-6)
    grid = np.linspace(x_min, x_max, GRID_POINTS)
    kde_pdf = kde(grid)

    rows = []
    plt.figure(figsize=(10, 5))
    bins = max(10, min(50, int(np.sqrt(len(cleaned)))))
    plt.hist(cleaned, bins=bins, density=True, alpha=0.3, label="Dati")
    plt.plot(grid, kde_pdf, color="black", label="KDE")

    for name, dist in DISTRIBUTIONS:
        try:
            if name in {"Lognormale", "Weibull", "Esponenziale"}:
                params = dist.fit(cleaned, floc=0)
            else:
                params = dist.fit(cleaned)
            pdf = dist.pdf(grid, *params)
            mse = float(np.mean((kde_pdf - pdf) ** 2))
            ks_pvalue, aic, bic = _ks_aic_bic(dist, params, cleaned)
            mean, std = _mean_std_from_params(dist, params)
            plausible = _plausible(mean, std)

            rows.append({
                "Distribuzione": name,
                "Parametri": "[" + ", ".join(f"{p:.6g}" for p in params) + "]",
                "MSE_KDE_PDF": mse,
                "KS_pvalue": ks_pvalue,
                "AIC": aic,
                "BIC": bic,
                "Mean": mean,
                "Std": std,
                "Plausible": plausible,
            })

            plt.plot(grid, pdf, label=name)
        except Exception as exc:
            logging.warning("Fit fallito per %s (%s): %s", label, name, exc)

    if not rows:
        logging.warning("Nessun fit riuscito per %s", label)
        return None

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_csv, index=False)
    logging.info("Statistiche fit salvate in %s", out_csv)

    plt.title(f"Confronto fit - {label}")
    plt.xlabel("Durata (giorni)")
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    logging.info("Grafico fit salvato in %s", out_png)

    return stats_df


def _choose_winner(stats_df):
    plausible = stats_df[stats_df["Plausible"]]
    if not plausible.empty:
        candidates = plausible
    else:
        candidates = stats_df
    return candidates.sort_values(by=["MSE_KDE_PDF", "AIC", "BIC"]).iloc[0]


def _to_fit_summary_row(stage, winner_row):
    params = winner_row["Parametri"].strip("[]").split(",")
    params = [float(p) for p in params if p.strip()]
    dist_name = winner_row["Distribuzione"]

    row = {
        "stage": stage,
        "dist": None,
        "s": np.nan,
        "c": np.nan,
        "loc": np.nan,
        "scale": np.nan,
        "mean": winner_row.get("Mean"),
        "std": winner_row.get("Std"),
        "mse": winner_row.get("MSE_KDE_PDF"),
    }

    if dist_name == "Lognormale":
        row["dist"] = "lognorm"
        if len(params) >= 3:
            row["s"], row["loc"], row["scale"] = params[:3]
    elif dist_name == "Weibull":
        row["dist"] = "weibull_min"
        if len(params) >= 3:
            row["c"], row["loc"], row["scale"] = params[:3]
    elif dist_name == "Esponenziale":
        row["dist"] = "expon"
        if len(params) >= 2:
            row["loc"], row["scale"] = params[:2]
    elif dist_name == "Normale":
        row["dist"] = "norm"
        if len(params) >= 2:
            row["loc"], row["scale"] = params[:2]

    return row


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

    summary_rows = []

    if "resolution_time_days" in df.columns:
        res_stats = _fit_distribution_set(
            df["resolution_time_days"],
            "resolution_time",
            OUT_CSV_DIR + "/distribution_fit_stats.csv",
            OUT_PNG_DIR + "/confronto_fit_resolution_time.png",
        )
        if res_stats is None:
            logging.warning("Fit risoluzione non disponibile.")
    else:
        logging.warning("Colonna resolution_time_days assente: skip fit risoluzione.")

    stage_defs = [
        ("development", "dev_duration_days"),
        ("review", "review_duration_days"),
        ("testing", "test_duration_days"),
    ]

    for stage, col in stage_defs:
        if col not in df.columns:
            logging.warning("Colonna fase assente: %s", col)
            continue
        stage_stats = _fit_distribution_set(
            df[col],
            stage,
            OUT_CSV_DIR + f"/distribution_fit_stats_{stage}.csv",
            OUT_PNG_DIR + f"/confronto_fit_{stage}.png",
        )
        if stage_stats is None:
            continue
        winner = _choose_winner(stage_stats)
        summary_rows.append(_to_fit_summary_row(stage, winner))

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUT_CSV_DIR + "/fit_summary.csv", index=False)
        logging.info("Fit summary salvato in %s", OUT_CSV_DIR + "/fit_summary.csv")
    else:
        logging.error("Nessun fit valido: fit_summary.csv non generato.")
