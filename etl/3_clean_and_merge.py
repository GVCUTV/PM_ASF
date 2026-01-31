# v7
# file: 3_clean_and_merge.py

"""
Pulizia/merge e DERIVAZIONE DELLE FASI con durate in GIORNI.
Allineato ai dataset prodotti da:
- 1_download_jira_tickets.py (v2)  -> jira_issues_raw.csv
- 2_download_github_prs.py  (v3)   -> github_prs_raw.csv  (endpoint /issues: NON garantisce 'merged_at')

Fasi derivate:
  development:  fields.created                     → first PR created_at
  review:       first PR created_at                → last PR merged_at (se presente) altrimenti last PR closed_at
  testing:      (review end)                       → fields.resolutiondate
In aggiunta: resolution_time_days = fields.resolutiondate − fields.created

NOTE: se mancano gli estremi, la durata corrispondente resta NaN (fail‑soft, nessuna supposizione).
"""

import pandas as pd
import logging
import re
import os
from path_config import PROJECT_ROOT

# === Logging setup ===
os.makedirs(PROJECT_ROOT+"/etl/output/logs", exist_ok=True)
os.makedirs(PROJECT_ROOT+"/etl/output/csv", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(PROJECT_ROOT+"/etl/output/logs/clean_and_merge.log"), logging.StreamHandler()]
)

# === Path I/O ===
JIRA_CSV = PROJECT_ROOT+"/etl/output/csv/jira_issues_raw.csv"
PRS_CSV = PROJECT_ROOT+"/etl/output/csv/github_prs_raw.csv"
OUT_TICKET_CSV = PROJECT_ROOT+"/etl/output/csv/jira_issues_clean.csv"
OUT_PR_CSV = PROJECT_ROOT+"/etl/output/csv/github_prs_clean.csv"
OUT_MERGE_CSV = PROJECT_ROOT+"/etl/output/csv/tickets_prs_merged.csv"  # con colonne di fase

# === Utility ===
def extract_jira_key(text):
    """Estrae la chiave Jira dal testo (es: BOOKKEEPER-1234)."""
    if pd.isnull(text):
        return None
    m = re.search(r'BOOKKEEPER-\d+', str(text))
    return m.group(0) if m else None

def to_datetime_safe(df, col):
    """Converte col a datetime (UTC→naive) con gestione errori; logga esito."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
        logging.info(f"Convertita a datetime (naive): {col}")
    else:
        logging.warning(f"Colonna assente: {col}")

def clean_tickets(df):
    """Pulizia base ticket JIRA (v2 downloader)."""
    before = len(df)
    df = df.drop_duplicates(subset=["key"])
    logging.info(f"Rimossi duplicati ticket per 'key': {before - len(df)}")

    bad_resolutions = ["Won't Fix", "Duplicate", "Not A Problem", "Incomplete", "Cannot Reproduce"]
    if "fields.resolution.name" in df.columns:
        before = len(df)
        df = df[~df['fields.resolution.name'].isin(bad_resolutions)]
        logging.info(f"Filtrati ticket con risoluzioni non utili: {before - len(df)}")
    else:
        logging.warning("Manca 'fields.resolution.name': nessun filtro su risoluzione.")

    # Timestamp principali da JIRA
    to_datetime_safe(df, "fields.created")
    to_datetime_safe(df, "fields.resolutiondate")
    # Alias comodi
    df["created"] = df.get("fields.created")
    df["resolved"] = df.get("fields.resolutiondate")
    return df

def clean_prs(df):
    """
    Pulizia PR (v3 downloader su /issues):
    - Estrae jira_key da title/body
    - Converte created_at/updated_at/closed_at a datetime
    - Se esistesse 'merged_at' (da altre fonti), lo converte; altrimenti resta mancante
    """
    # Estrazione chiave JIRA
    if "title" in df.columns:
        df["jira_key"] = df["title"].apply(extract_jira_key)
    else:
        df["jira_key"] = None
    if df["jira_key"].isna().all() and "body" in df.columns:
        df["jira_key"] = df["body"].apply(extract_jira_key)

    # Timestamp tipici dell'endpoint /issues
    for col in ["created_at", "updated_at", "closed_at"]:
        to_datetime_safe(df, col)

    # In genere /issues NON fornisce merged_at. Se c'è, lo gestiamo.
    if "merged_at" in df.columns:
        to_datetime_safe(df, "merged_at")
    else:
        logging.info("Campo 'merged_at' NON presente: useremo 'closed_at' come proxy di fine review se necessario.")

    return df

def derive_phase_times(merged):
    """
    Deriva i timestamp per fase usando aggregazioni per ticket 'key'.
    Evita errori GroupBy uscendo da selezioni multiple: aggregazioni colonnari indipendenti.
    """
    logging.info("Calcolo aggregazioni PR per ticket (first created_at, last merged/closed)...")

    # Default colonne di aggregazione assenti
    merged["first_pr_created_at"] = pd.NaT
    merged["last_pr_merged_at"] = pd.NaT
    merged["last_pr_closed_at"] = pd.NaT

    if "key" not in merged.columns:
        logging.warning("Colonna 'key' mancante nel merge: impossibile aggregare per ticket.")
    else:
        grp = merged.groupby("key", dropna=False)

        # Min created_at per ticket (inizio review = fine dev)
        if "created_at" in merged.columns:
            try:
                first_created = grp["created_at"].min()
                merged["first_pr_created_at"] = merged["key"].map(first_created)
            except Exception as e:
                logging.warning(f"Min created_at per ticket fallito: {e}")

        # Max merged_at e closed_at per ticket
        if "merged_at" in merged.columns:
            try:
                last_merged = grp["merged_at"].max()
                merged["last_pr_merged_at"] = merged["key"].map(last_merged)
            except Exception as e:
                logging.warning(f"Max merged_at per ticket fallito: {e}")

        if "closed_at" in merged.columns:
            try:
                last_closed = grp["closed_at"].max()
                merged["last_pr_closed_at"] = merged["key"].map(last_closed)
            except Exception as e:
                logging.warning(f"Max closed_at per ticket fallito: {e}")

    # Timestamp fasi
    merged["dev_start_ts"] = merged.get("fields.created")
    merged["dev_end_ts"] = merged["first_pr_created_at"]

    merged["review_start_ts"] = merged["dev_end_ts"]
    # Fine review: preferisci merged_at se disponibile, altrimenti closed_at
    merged["review_end_ts"] = merged["last_pr_merged_at"].fillna(merged["last_pr_closed_at"])

    merged["test_start_ts"] = merged["review_end_ts"]
    merged["test_end_ts"] = merged.get("fields.resolutiondate")

    # Durata in giorni
    def duration_days(start, end):
        if pd.isna(start) or pd.isna(end):
            return pd.NA
        delta = (end - start).total_seconds() / 86400.0
        return delta if delta >= 0 else pd.NA

    logging.info("Calcolo durate (giorni) per ciascuna fase...")
    merged["dev_duration_days"] = [duration_days(s, e) for s, e in zip(merged["dev_start_ts"], merged["dev_end_ts"])]
    merged["review_duration_days"] = [duration_days(s, e) for s, e in zip(merged["review_start_ts"], merged["review_end_ts"])]
    merged["test_duration_days"] = [duration_days(s, e) for s, e in zip(merged["test_start_ts"], merged["test_end_ts"])]

    # Durata totale (creazione → risoluzione) in giorni
    if "resolution_time_days" not in merged.columns and \
       "fields.created" in merged.columns and "fields.resolutiondate" in merged.columns:
        merged["resolution_time_days"] = (merged["fields.resolutiondate"] - merged["fields.created"]).dt.total_seconds() / 86400.0

    # Log diagnostici
    for col in ["dev_duration_days", "review_duration_days", "test_duration_days", "resolution_time_days"]:
        if col in merged.columns:
            valid = pd.to_numeric(merged[col], errors="coerce").dropna()
            if valid.size:
                logging.info(f"[{col}] validi={valid.size} | mean={valid.mean():.2f} | median={valid.median():.2f}")
            else:
                logging.info(f"[{col}] nessun valore valido.")

    return merged

if __name__ == "__main__":
    # Ticket
    tickets = pd.read_csv(JIRA_CSV, low_memory=False)
    logging.info(f"Caricati ticket raw: {JIRA_CSV} ({len(tickets)})")
    tickets_clean = clean_tickets(tickets)
    tickets_clean.to_csv(OUT_TICKET_CSV, index=False)
    logging.info(f"Salvati {len(tickets_clean)} ticket puliti in {OUT_TICKET_CSV}")

    # PRs
    prs = pd.read_csv(PRS_CSV, low_memory=False)
    logging.info(f"Caricate PR raw: {PRS_CSV} ({len(prs)})")
    prs_clean = clean_prs(prs)
    prs_clean.to_csv(OUT_PR_CSV, index=False)
    logging.info(f"Salvate {len(prs_clean)} PR pulite in {OUT_PR_CSV}")

    # Merge per key
    merged = pd.merge(tickets_clean, prs_clean, left_on="key", right_on="jira_key", how="left",
                      suffixes=("_ticket", "_pr"))
    logging.info(f"Merge completato: righe={len(merged)}. Derivo fasi...")
    merged = derive_phase_times(merged)

    # Output finale
    merged.to_csv(OUT_MERGE_CSV, index=False)
    logging.info(f"Salvati dati mergeati (ticket+PR+fasi) in {OUT_MERGE_CSV}")
