# v3
# file: 4_summarize_and_plot.py

"""
Calcola statistiche base, stampa tabella, mostra grafici.
Questa versione genera il grafico a torta senza etichette sovrapposte: la legenda visualizza la corrispondenza colore/tipo issue.
"""

import matplotlib
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os
from path_config import PROJECT_ROOT

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT+"/etl/output/logs/summarize_and_plot.log"),
        logging.StreamHandler()
    ]
)

IN_CSV = PROJECT_ROOT+"/etl/output/csv/tickets_prs_merged.csv"

if __name__ == "__main__":
    # Carica i dati dal CSV
    df = pd.read_csv(IN_CSV)

    # Numero totale ticket
    total = len(df)
    logging.info(f"Ticket totali: {total}")

    # Suddivisione per tipo
    type_counts = df['fields.issuetype.name'].value_counts()
    print("Suddivisione per tipo:\n", type_counts)

    # Ticket riaperti
    reopened = df[df['fields.status.name'].str.contains("Reopen", na=False)]
    logging.info(f"Ticket riaperti: {len(reopened)} ({len(reopened) / total * 100:.1f}%)")

    # Ticket in progress
    in_progress = df[df['fields.status.name'] == "In Progress"]
    logging.info(f"In Progress: {len(in_progress)} ({len(in_progress) / total * 100:.1f}%)")

    # Ticket chiusi senza PR
    closed_no_pr = df[
        (df['fields.status.name'].isin(["Closed", "Resolved"])) &
        (df['jira_key'].isnull())
        ]
    logging.info(f"Ticket chiusi senza PR: {len(closed_no_pr)} ({len(closed_no_pr) / total * 100:.1f}%)")

    # Tabella finale
    summary = pd.DataFrame({
        "Tipo": type_counts.index,
        "Numero": type_counts.values,
        "% Totale": [f"{x / total * 100:.1f}%" for x in type_counts.values]
    })
    print("\nTabella riassuntiva:\n", summary.to_string(index=False))

    # -- Grafico a torta con legenda (NO etichette sovrapposte) --
    os.makedirs(PROJECT_ROOT+"/etl/output/png", exist_ok=True)
    plt.figure(figsize=(8, 8))
    wedges, texts = plt.pie(
        type_counts,
        labels=None,  # Nessuna etichetta diretta sulla torta!
        autopct=None,
        startangle=90
    )
    # Costruisci la legenda con tipi e valori %
    labels_legend = [
        f"{t} ({v} - {v / total:.1%})"
        for t, v in zip(type_counts.index, type_counts.values)
    ]
    plt.legend(
        wedges,
        labels_legend,
        title="Tipo di Ticket",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )
    plt.title("Distribuzione Ticket per Tipo")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Lascio spazio a destra per la legenda
    plt.savefig(PROJECT_ROOT+"/etl/output/png/distribuzione_ticket_tipo.png")
    logging.info("Grafico a torta salvato in ./output/png/distribuzione_ticket_tipo.png")
    # plt.show()  # Non necessario/headless

    # Export summary
    summary.to_csv(PROJECT_ROOT+"/etl/output/csv/statistiche_riassuntive.csv", index=False)
    logging.info("Statistiche esportate in ./output/statistiche_riassuntive.csv")

"""
Note:
- La legenda mostra tipo, conteggio e percentuale.
- Nessuna etichetta scritta direttamente sulla torta: massimo ordine.
- Il grafico è già pronto per presentazioni e report.
"""
