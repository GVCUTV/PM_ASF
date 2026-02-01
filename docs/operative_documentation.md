<!-- v1 -->
<!-- filename: operative_documentation.md -->

# Operative Documentation

## ETL Overview (Jira + GitHub Ingestion to Raw CSVs)

**What it does**
- The ETL pipeline in `etl/` is a sequence of stand-alone scripts that download raw Jira issues and GitHub pull request data into CSVs, which are later cleaned and merged by downstream steps. This section documents the ingestion layer that produces the raw datasets used by later ETL stages.

**How it is implemented**
- The ingestion layer is composed of two scripts invoked manually from the repo root: `etl/1_download_jira_tickets.py` and `etl/2_download_github_prs.py`.
- Both scripts derive output locations from `path_config.PROJECT_ROOT` and write raw CSVs under `etl/output/csv/` plus logs under `etl/output/logs/`.
- Each script is responsible for contacting an external API (Jira or GitHub), paginating results, flattening/deriving key fields, and persisting output to disk.

**How to use it**
1. Ensure you have network access and Python dependencies available (`requests`, `pandas`, etc.).
2. Run the scripts from the repository root:
   - `python etl/1_download_jira_tickets.py`
   - `python etl/2_download_github_prs.py`
3. Verify the raw CSV outputs:
   - `etl/output/csv/jira_issues_raw.csv`
   - `etl/output/csv/github_prs_raw.csv`
4. Proceed with downstream steps (e.g., `etl/3_clean_and_merge.py`) to clean and merge these raw datasets.

**Outputs and contracts**
- `jira_issues_raw.csv` is the canonical raw Jira issue export (flattened with dot-separated fields).
- `github_prs_raw.csv` is the canonical raw PR export (including review/CI details and derived summary fields).
- Re-running the scripts overwrites existing outputs and updates logs; results depend on live API data and any incremental/caching behavior described below.

---

## ETL Script: `etl/1_download_jira_tickets.py`

**What it does**
- Downloads Jira issues for the BOOKKEEPER project using the Jira REST API, flattens nested fields, and writes a raw CSV for downstream cleaning and merges.
- Logs counts and basic assignee coverage for diagnostics.

**How it is implemented**
- Uses a paginated Jira search (`/rest/api/2/search`) with a fixed JQL query and `startAt`/`maxResults` pagination.
- Implements a retry loop with backoff for transient failures (e.g., 429/5xx).
- Flattens the response with `pandas.json_normalize` and prefers a known subset of Jira fields. If those columns are missing, it falls back to writing all flattened columns.
- Ensures that even an empty result still produces a CSV with headers, allowing downstream steps to run without file-not-found errors.

**How it must be used**
- **Command:** `python etl/1_download_jira_tickets.py`
- **Inputs:** Jira REST API (network); no local input files.
- **Outputs:**
  - `etl/output/csv/jira_issues_raw.csv` (flattened Jira issues)
  - `etl/output/logs/download_jira_tickets.log`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`. Jira access is anonymous; no token configuration is required.
- **Limitations:** The script requests a fixed subset of fields and caps pagination by an internal batch limit, so extremely large issue sets may require adjusting that logic.

---

## ETL Script: `etl/2_download_github_prs.py`

**What it does**
- Downloads GitHub pull requests for `apache/bookkeeper`, including per-PR details (reviews, check-runs, and combined statuses), and writes a raw CSV for downstream cleaning and merges.
- Supports incremental reuse of unchanged PR rows, ETag caching, multi-token rotation, and concurrency to reduce API load.

**How it is implemented**
- Lists PR pages concurrently and uses a cached HTTP layer (ETags) to avoid re-downloading unchanged data.
- Optionally reuses prior CSV rows when `updated_at` is unchanged to avoid per-PR detail requests.
- For each PR needing refresh, it fetches reviews, check-runs, and combined statuses in parallel and derives summary fields (review/CI signals).
- Manages API rate limits via token rotation and backoff/sleep strategies on 403/429 responses.

**How it must be used**
- **Command:** `python etl/2_download_github_prs.py`
- **Inputs:** GitHub REST API (network). Tokens are read from `etl/env/github_tokens.env` and/or `GITHUB_TOKENS` env var (optional but strongly recommended).
- **Outputs:**
  - `etl/output/csv/github_prs_raw.csv` (raw PR rows with derived review/CI fields)
  - `etl/output/logs/download_github_prs.log`
  - `etl/cache/github_http_cache.json` (ETag cache)
- **Configuration:**
  - Token pool uses `GITHUB_TOKENS` or `etl/env/github_tokens.env`.
  - Incremental mode reuses rows based on `updated_at` (if a prior CSV exists).
- **Limitations:** Running without tokens falls back to anonymous access with low rate limits; large repos may take a long time or yield partial data.

---

## ETL Overview (Raw CSVs to Merged Dataset and Summaries)

**What it does**
- The downstream ETL steps transform raw Jira/GitHub CSVs into a merged, ticket-centric dataset with derived phase timestamps/durations, then produce high-level summary outputs for reporting.
- The key contract produced by this stage is `etl/output/csv/tickets_prs_merged.csv`, which is consumed by analysis scripts that estimate parameters, fit distributions, and diagnose data quality.

**How it is implemented**
- The workflow is not orchestrated by a single runner; scripts are executed manually in order after the ingestion layer.
- `etl/3_clean_and_merge.py` loads raw Jira/PR exports, cleans and normalizes them, merges on Jira key, and derives development/review/testing timestamps and duration columns.
- `etl/4_summarize_and_plot.py` reads the merged dataset to generate summary counts and a ticket-type pie chart for quick reporting.

**How to use it**
1. Ensure the raw CSVs exist (`etl/output/csv/jira_issues_raw.csv`, `etl/output/csv/github_prs_raw.csv`).
2. Run `etl/3_clean_and_merge.py` to generate the cleaned and merged outputs.
3. Run `etl/4_summarize_and_plot.py` to produce summary tables and a pie chart.
4. Use the merged CSV as input to downstream analysis scripts (parameter estimation, distribution fitting, enrichment).

**Outputs and contracts**
- `etl/output/csv/jira_issues_clean.csv` and `etl/output/csv/github_prs_clean.csv` are intermediate cleaned datasets.
- `etl/output/csv/tickets_prs_merged.csv` is the canonical merged dataset used by later ETL steps.
- `etl/output/csv/statistiche_riassuntive.csv` and `etl/output/png/distribuzione_ticket_tipo.png` are summary artifacts intended for reporting.

---

## ETL Script: `etl/3_clean_and_merge.py`

**What it does**
- Cleans raw Jira issues and GitHub PR exports, merges them on Jira key, and derives phase timestamps/durations for development, review, and testing.
- Produces the canonical merged dataset (`tickets_prs_merged.csv`) consumed by downstream ETL analysis and reporting scripts.

**How it is implemented**
- Loads `jira_issues_raw.csv` and `github_prs_raw.csv` from `etl/output/csv/`.
- Cleans Jira issues by de-duplicating on Jira key, filtering out specific resolution values, and normalizing timestamps.
- Cleans PR data by extracting Jira keys from title/body using a regex and normalizing PR timestamps; it uses `merged_at` when present and falls back to `closed_at` for review end.
- Merges the datasets with a left join on Jira key, then aggregates PR timestamps to compute phase start/end points and duration columns (`dev_duration_days`, `review_duration_days`, `test_duration_days`).
- Writes cleaned CSVs for both Jira and PRs plus the merged dataset and logs the run.

**How it must be used**
- **Command:** `python etl/3_clean_and_merge.py`
- **Inputs:**
  - `etl/output/csv/jira_issues_raw.csv`
  - `etl/output/csv/github_prs_raw.csv`
- **Outputs:**
  - `etl/output/csv/jira_issues_clean.csv`
  - `etl/output/csv/github_prs_clean.csv`
  - `etl/output/csv/tickets_prs_merged.csv`
  - `etl/output/logs/clean_and_merge.log`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`; Jira key extraction is hard-coded to `BOOKKEEPER-<num>` regex.
- **Limitations:**
  - If required columns are missing, the script logs warnings and leaves derived fields as `NaN`.
  - Missing or negative phase deltas are set to `NaN` to avoid invalid durations.
  - Resolution filtering can remove tickets and affect totals; it is not currently configurable.

---

## ETL Script: `etl/4_summarize_and_plot.py`

**What it does**
- Produces summary counts by issue type and generates a ticket-type pie chart from the merged dataset.
- Exports the summary table to CSV for reporting.

**How it is implemented**
- Loads `etl/output/csv/tickets_prs_merged.csv` and groups rows by `fields.issuetype.name` to compute counts.
- Identifies basic ticket status categories (reopened, in-progress, closed without PR) using columns such as `fields.status.name` and PR linkage.
- Generates a pie chart with a legend (to avoid label overlap) and saves it as a PNG.
- Writes the summary counts to `statistiche_riassuntive.csv` and logs the run.

**How it must be used**
- **Command:** `python etl/4_summarize_and_plot.py`
- **Inputs:** `etl/output/csv/tickets_prs_merged.csv`
- **Outputs:**
  - `etl/output/png/distribuzione_ticket_tipo.png`
  - `etl/output/csv/statistiche_riassuntive.csv`
  - `etl/output/logs/summarize_and_plot.log`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`.
- **Limitations:** Assumes columns like `fields.issuetype.name` and `fields.status.name` exist; missing columns will raise errors at runtime.

---

## ETL Script: `etl/5_estimate_parameters.py`

**What it does**
- Computes global arrival/throughput metrics and resolution-time summaries from the merged ticket dataset.
- Exports per-phase duration tables and summary statistics for development, review, and testing.
- Builds a daily backlog time series (open tickets) and saves a PNG plot.

**How it is implemented**
- Loads `etl/output/csv/tickets_prs_merged.csv` and normalizes timestamps for creation/resolution.
- Estimates inter-arrival times (days) from sorted creation timestamps and derives an arrival rate as `1 / mean_interarrival`.
- Computes global resolution-time mean/median (in days) and a mean monthly throughput based on resolution dates.
- Summarizes phase duration columns (`dev_duration_days`, `review_duration_days`, `test_duration_days`) with counts, missing-share, mean, median, std, quartiles, and min/max.
- Calculates backlog using cumulative daily created vs resolved counts and plots the series.

**How it must be used**
- **Command:** `python etl/5_estimate_parameters.py`
- **Inputs:** `etl/output/csv/tickets_prs_merged.csv`
- **Outputs:**
  - `etl/output/csv/parameter_estimates.csv`
  - `etl/output/csv/phase_durations_wide.csv`
  - `etl/output/csv/phase_summary_stats.csv`
  - `etl/output/png/backlog_over_time.png`
  - `etl/output/logs/estimate_parameters.log`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`.
- **Limitations:** If required timestamp or phase columns are missing, the script logs warnings and skips or leaves related metrics as `NaN`.

---

## ETL Script: `etl/6_diagnose_and_plot_tickets.py`

**What it does**
- Provides per-ticket diagnostics to surface data quality issues in the merged dataset.
- Computes resolution time in hours (when possible) and plots its distribution between 0 and 10,000 hours.

**How it is implemented**
- Loads `etl/output/csv/tickets_prs_merged.csv` and resolves key columns for ticket ID, creation date, resolution date, status, and issue type.
- Converts creation/resolution timestamps to timezone-naive datetimes and derives `resolution_time_hours` either from those timestamps or from `resolution_time_days` when timestamps are unavailable.
- Iterates each ticket to print a diagnostic line and flags common inconsistencies (missing key, missing creation, resolution before creation, closed without resolution date, negative resolution time).
- Builds a histogram of valid resolution times (0–10,000 hours) with bin count scaled to sample size and saves the plot.

**How it must be used**
- **Command:** `python etl/6_diagnose_and_plot_tickets.py`
- **Inputs:** `etl/output/csv/tickets_prs_merged.csv`
- **Outputs:**
  - `etl/output/png/distribuzione_resolution_times_0_10000.png`
  - `etl/output/logs/diagnose_tickets.log`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`.
- **Limitations:** The script prints per-ticket diagnostics to stdout, which can be verbose on large datasets; histogram generation is skipped when no valid resolution-time data is available.

---

## ETL Script: `etl/3_phase_duration_distribution_etl.py` — Distribution Parameters Output

**What it does**
- Extends the phase-duration distribution summary output to include the fitted distribution parameters for each phase (dev/review/testing).
- Records the parameter values in a JSON-formatted string in the `parameters` column of `etl/output/csv/distribution_summary.csv`, alongside the `best_fit` distribution name.

**How it is implemented**
- Fits exponential (rate), lognormal (mu, sigma), and Weibull (shape, scale) distributions per phase using the existing log-likelihood estimators.
- Selects the best-fit distribution by maximum log-likelihood, then serializes the parameters for that best-fit distribution into a JSON object.
- Writes the JSON string to the `parameters` column immediately after the `best_fit` column in the CSV schema.

**How it must be used**
- **Command:** `python etl/3_phase_duration_distribution_etl.py`
- **Inputs:**
  - `etl/output/csv/jira_issues_raw.csv` or `etl/output/csv/jira_tickets_raw.csv`
  - `etl/output/csv/github_prs_raw.csv`
- **Outputs:**
  - `etl/output/csv/distribution_summary.csv` (now includes `parameters` column with JSON-formatted fitted parameters)
  - `etl/output/csv/distribution_summary.md`
  - `etl/output/phase_durations.csv`
- **Limitations:** If there are insufficient or invalid samples for a phase, the `parameters` column contains an empty JSON object (`{}`) for that phase.

---

## ETL Script: `etl/7_fit_distributions.py`

**What it does**
- Fit candidate distributions (lognormal, Weibull, exponential, normal) to resolution time and to per-phase durations (development/review/testing).
- Produces per-stage fit statistics CSVs, comparison plots against KDE, and a compact `fit_summary.csv` for simulation configuration.

**How it is implemented**
- Loads the merged dataset (`etl/output/csv/tickets_prs_merged.csv`) and computes `resolution_time_days` if it is missing by subtracting creation and resolution timestamps.
- Filters duration samples to non-negative finite values capped at 10 years (`MAX_DAYS`) and requires a minimum sample size before fitting.
- Uses KDE as a reference density and fits each candidate distribution with SciPy; for each fit it computes MSE vs KDE, KS p-value, AIC/BIC, mean/std, and a plausibility flag.
- Writes per-stage statistics to `distribution_fit_stats_<stage>.csv`, saves comparison plots `confronto_fit_<stage>.png`, and selects the best fit (by minimum MSE, with plausibility preference) to populate `fit_summary.csv` with SciPy-compatible parameter fields.

**How it must be used**
- **Command:** `python etl/7_fit_distributions.py`
- **Inputs:** `etl/output/csv/tickets_prs_merged.csv` (from `etl/3_clean_and_merge.py`).
- **Outputs:**
  - `etl/output/csv/distribution_fit_stats.csv` (resolution-time fits)
  - `etl/output/csv/distribution_fit_stats_development.csv`
  - `etl/output/csv/distribution_fit_stats_review.csv`
  - `etl/output/csv/distribution_fit_stats_testing.csv`
  - `etl/output/csv/fit_summary.csv`
  - `etl/output/png/confronto_fit_resolution_time.png`
  - `etl/output/png/confronto_fit_development.png`
  - `etl/output/png/confronto_fit_review.png`
  - `etl/output/png/confronto_fit_testing.png`
  - `etl/output/logs/fit_distributions.log`
- **Limitations:** Requires at least 10 valid samples per series; if KDE or fits fail, it skips output for that series and logs a warning.

---

## ETL Script: `etl/8_export_fit_summary.py`

**What it does**
- Converts per-stage distribution fit statistics into a compact `fit_summary.csv` for simulation inputs.
- Selects the best fit per stage using the available error metric (MSE/MAE) and optional plausibility filtering.

**How it is implemented**
- Loads `distribution_fit_stats_<stage>.csv` for each requested stage, requiring `Distribuzione` and `Parametri` columns.
- Parses the `Parametri` string into floats and maps Italian distribution labels to SciPy names.
- Chooses the best fit by sorting on the first available metric (MSE/MAE) with AIC/BIC tie-breakers.
- Writes `fit_summary.csv` with normalized columns (`stage`, `dist`, `s`, `c`, `loc`, `scale`, `mean`, `std`, `mse`).

**How it must be used**
- **Command:** `python etl/8_export_fit_summary.py --stages development review testing`
- **Inputs:** `etl/output/csv/distribution_fit_stats_<stage>.csv` for each stage (or overrides via `--stage-csv`).
- **Outputs:** `etl/output/csv/fit_summary.csv`, `etl/output/logs/export_fit_summary.log`.
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`; use `--base-dir` and `--out-csv` to override locations.
- **Limitations:** The script fails fast if input CSVs are missing required columns or if no metric column is found.

---

## ETL Script: `etl/9_enrich_feedback_cols.py`

**What it does**
- Arricchisce il dataset `tickets_prs_merged.csv` con segnali di feedback review/CI e ruoli stimati (developer/tester).
- Deriva colonne aggiuntive utili per analisi del rework e feedback loop (review rounds, richieste modifiche, esito CI).

**How it is implemented**
- Carica il merged CSV e calcola `review_rounds` usando `reviews_count` quando disponibile o il numero di stati in `pull_request_review_states`.
- Imposta `review_rework_flag` quando `requested_changes_count` è maggiore di zero o quando lo stato review include `CHANGES_REQUESTED`.
- Interpreta le liste JSON di `check_runs_conclusions` e/o `combined_status_states` per stimare `ci_failed_then_fix` (fallimento seguito da successo).
- Deduce `dev_user` da colonne prioritizzate (autore PR, assegnatario, assignee Jira) e `tester` da reviewer richiesti o reporter Jira.
- Registra log di copertura per le colonne arricchite e salva il CSV di output.

**How it must be used**
- **Command:** `python etl/9_enrich_feedback_cols.py`
- **Inputs:** `etl/output/csv/tickets_prs_merged.csv` (da `etl/3_clean_and_merge.py`).
- **Outputs:**
  - `etl/output/csv/tickets_prs_merged.csv` (default, sovrascrive l’input) oppure un percorso specificato con `--out-csv`.
  - `etl/output/logs/enrich_feedback.log`
- **Configuration:** Path risolti con `path_config.PROJECT_ROOT`; `--in-csv`, `--out-csv`, `--log-path` per override.
- **Limitations:** I segnali derivati sono euristici e dipendono dalla presenza delle colonne review/CI nel dataset; se assenti, le colonne possono restare vuote o non essere prodotte.

---

## ETL Script: `etl/10_phase_duration_distribution_etl.py`

**What it does**
- Computes per-ticket phase boundary timestamps and durations (development, review, testing) directly from the raw Jira/GitHub exports.
- Produces ticket-level duration outputs plus distribution summaries and histogram plots for each phase.

**How it is implemented**
- Loads Jira raw exports from `etl/output/csv/jira_tickets_raw.csv` (preferred) or falls back to `etl/output/csv/jira_issues_raw.csv`.
- Loads GitHub PR exports from `etl/output/csv/github_prs_raw.csv`, extracts Jira keys with the `BOOKKEEPER-\d+` regex, and aggregates PR timestamps to derive review boundaries.
- Normalizes all timestamps to UTC and derives phase boundaries according to the source-of-truth fallback rules:
  - Dev start = Jira created timestamp.
  - Review start = earliest PR created timestamp linked to the ticket.
  - Review end = latest PR merged timestamp (falls back to PR close when merge is missing).
  - Testing end = Jira resolution timestamp (falls back to review end if missing).
- Calculates phase durations in hours and records exception reasons when boundaries are missing or invalid.
- Computes distribution summaries (count, mean, median, variance, percentiles) and selects the best-fit distribution by log-likelihood among exponential, log-normal, and Weibull.
- Generates histogram PNGs using a lightweight built-in renderer to avoid adding new dependencies.

**How it must be used**
- **Command:** `python etl/10_phase_duration_distribution_etl.py`
- **Inputs:**
  - `etl/output/csv/github_prs_raw.csv`
  - `etl/output/csv/jira_tickets_raw.csv` (preferred) or `etl/output/csv/jira_issues_raw.csv` (fallback)
- **Outputs:**
  - `phase_durations.csv`
  - `distribution_summary.csv`
  - `distribution_summary.md`
  - `plots/dev_phase_histogram.png`
  - `plots/review_phase_histogram.png`
  - `plots/testing_phase_histogram.png`
- **Limitations:** Jira transition history is not consumed by this script; when transitions are unavailable the markdown summary prompts for user confirmation of fallback boundary inference.

---

## ETL Script: `etl/state_parameters.py`

**What it does**
- Builds per-developer state transitions and duration PMFs for the workflow states **OFF**, **DEV**, **REV**, and **TEST**.
- Reads GitHub PR assignees, joins them to Jira phase timestamps, expands per-developer events, and outputs a transition matrix plus per-state PMFs for simulation parameterization.

**How it is implemented**
- Loads `etl/output/csv/github_prs_raw.csv` and extracts assignee identities from the `assignees` column.
- Loads `etl/output/csv/phase_durations.csv`, parses DEV/REV/TEST timestamps, and drops rows missing any required phase timestamp before event construction.
- Extracts Jira keys from PR titles/bodies, maps them to phase timestamps, and constructs DEV/REV/TEST events with start/end times derived from the phase columns.
- For each developer, events are sorted by start time; transitions are built with explicit `OFF → first_state`, gap-driven `state → OFF → next_state`, and final `state → OFF` transitions.
- Collects positive-duration stints for each state (DEV/REV/TEST from event durations, OFF from idle gaps), rounds durations to 1e-3 days, counts occurrences, and normalizes to a PMF.
- Emits a normalized transition matrix (rows/cols labeled `OFF,DEV,REV,TEST`) and one PMF CSV per state under `data/state_parameters/`.

**How it must be used**
- **Command:** `python etl/state_parameters.py`
- **Inputs:**
  - `etl/output/csv/github_prs_raw.csv`
  - `etl/output/csv/phase_durations.csv`
  - `etl/output/csv/distribution_summary.csv` (presence check only)
- **Outputs:**
  - `data/state_parameters/transition_matrix.csv`
  - `data/state_parameters/pmf_off.csv`
  - `data/state_parameters/pmf_dev.csv`
  - `data/state_parameters/pmf_rev.csv`
  - `data/state_parameters/pmf_test.csv`
- **Validation:** The script checks that each transition-matrix row and each PMF sums to ~1.0 within a configurable tolerance.
- **Limitations:** PRs without Jira keys or assignees are skipped; phase rows missing any required timestamps are excluded from event generation.
