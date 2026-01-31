# v3
# file: etl/1_download_jira_tickets.py
"""
Downloader Jira per Apache BookKeeper (BK) con flatten corretto dei campi annidati
(es. assignee) e logging dettagliato di ogni operazione.

Cosa fa:
- Scarica tutti i ticket BK via API di Jira (paginazione).
- Seleziona i campi utili e li flattens con sep='.' così da ottenere
  colonne come 'fields.assignee.name' e 'fields.assignee.displayName'.
- Scrive un CSV pronto per i passi ETL successivi.

Repo: https://github.com/GVCUTV/BK_ASF.git
"""

from __future__ import print_function

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from os import path

# --------------------------- Config & logging --------------------------- #

from path_config import PROJECT_ROOT  # CWD-indipendente (come richiesto)

LOG_DIR = path.join(PROJECT_ROOT, "output", "logs")
OUT_CSV = path.join(PROJECT_ROOT, "etl", "output", "csv", "jira_issues_raw.csv")

JIRA_DOMAIN = "https://issues.apache.org/jira"
PROJECT_KEY = "BOOKKEEPER"
JQL = "project = {proj} ORDER BY created ASC".format(proj=PROJECT_KEY)

FIELDS = ",".join([
    "key",
    "summary",
    "issuetype",
    "status",
    "resolution",
    "resolutiondate",
    "created",
    "updated",
    "assignee",          # oggetto; lo flatteniamo in subfields
    "description",
])

MAX_RESULTS = 1000      # limite Jira per call
MAX_BATCHES = 200       # sicurezza
RETRY_SLEEP = 5.0       # sec tra retry

def _safe_mkdirs(d):
    try:
        os.makedirs(d)
    except OSError:
        if not path.isdir(d):
            raise

def _setup_logging():
    _safe_mkdirs(LOG_DIR)
    log_path = path.join(LOG_DIR, "download_jira_tickets.log")

    root = logging.getLogger()
    root.handlers[:] = []
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

    logging.info("Logger pronto. Logfile: %s", log_path)
    return log_path


# --------------------------- Jira API helpers --------------------------- #

def _jira_get(url, params=None, retries=3):
    """GET con retry semplice e logging."""
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            logging.warning("Jira GET %s status=%s params=%s", url, r.status_code, params)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(RETRY_SLEEP * (i + 1))
            else:
                break
        except requests.RequestException as e:
            logging.warning("Jira exception: %s (attempt %d)", e, i + 1)
            time.sleep(RETRY_SLEEP * (i + 1))
    raise RuntimeError("Jira GET failed after retries: %s" % url)


def download_all_issues(jql, fields, max_results=MAX_RESULTS, max_batches=MAX_BATCHES):
    """Scarica tutte le issue con paginazione."""
    base_url = JIRA_DOMAIN + "/rest/api/2/search"
    issues = []
    start_at = 0
    total = None
    for b in range(max_batches):
        params = {
            "jql": jql,
            "fields": fields,
            "startAt": start_at,
            "maxResults": max_results,
            "expand": "changelog",
        }
        logging.info("Jira batch %d: startAt=%d", b + 1, start_at)
        data = _jira_get(base_url, params=params)
        if total is None:
            total = data.get("total", 0)
            logging.info("Jira total issues: %d", total)

        page = data.get("issues", []) or []
        logging.info("Jira page size: %d", len(page))
        issues.extend(page)

        if len(page) < max_results:
            break
        start_at += max_results

    logging.info("Downloaded issues: %d", len(issues))
    return issues


# --------------------------- Main --------------------------- #

def main():
    _setup_logging()
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("OUTPUT CSV   : %s", OUT_CSV)

    issues = download_all_issues(JQL, FIELDS)

    if not issues:
        logging.warning("Nessuna issue scaricata.")
        # comunque crea file vuoto con header minimo
        _safe_mkdirs(path.dirname(OUT_CSV))
        pd.DataFrame([]).to_csv(OUT_CSV, index=False)
        return

    # Flatten con sep='.' per ottenere subfields come 'fields.assignee.name'
    df = pd.json_normalize(issues, sep='.')
    logging.info("Flattened dataframe shape: %s", df.shape)

    # Selezione colonne chiave + subfields di assignee
    prefer = [
        "key",
        "fields.summary",
        "fields.issuetype.name", "fields.issuetype.id",
        "fields.status.name",    "fields.status.id",
        "fields.resolution.name", "fields.resolutiondate",
        "fields.created", "fields.updated",
        "fields.assignee.name", "fields.assignee.displayName", "fields.assignee.key",
        "fields.description",
    ]
    cols = [c for c in prefer if c in df.columns]
    if cols:
        df = df[cols]
    else:
        logging.warning("Colonne preferite non trovate, salvo tutte le colonne flattenate.")

    # Salva CSV
    _safe_mkdirs(path.dirname(OUT_CSV))
    df.to_csv(OUT_CSV, index=False)
    logging.info("Salvate %d righe in %s", len(df), OUT_CSV)

    # Copertura assignee per debug
    for c in ["fields.assignee.name", "fields.assignee.displayName"]:
        if c in df.columns:
            nonnull = int(df[c].notna().sum())
            distinct = int(df[c].dropna().astype(str).str.strip().nunique())
            logging.info("%s non-null=%d distinct=%d", c, nonnull, distinct)


if __name__ == "__main__":
    main()
