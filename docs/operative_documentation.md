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

## ETL Script: `etl/5_arrival_rate.py`

**What it does**
- Computes the arrival rate of Jira issues (tickets) per day from the raw Jira export CSV.
- Writes a one-row CSV containing the count of tickets, observation window, span in days, and arrival rate.

**How it is implemented**
- Reads `etl/output/csv/jira_issues_raw.csv` (falling back to `jira_tickets_raw.csv` if needed) and parses `fields.created` timestamps.
- Uses the earliest and latest creation timestamps to compute the observation window length in days.
- Calculates arrival rate as `ticket_count / span_days` and writes the output to `etl/output/csv/arrival_rate_jira_issues.csv`.

**How it must be used**
- **Command:** `python etl/5_arrival_rate.py`
- **Inputs:** `etl/output/csv/jira_issues_raw.csv` (raw Jira export).
- **Outputs:** `etl/output/csv/arrival_rate_jira_issues.csv`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`.
- **Limitations:** Requires at least two Jira issues with valid `fields.created` timestamps to compute a non-zero span.

---

## State-Parameter Extraction: `etl/state_parameters.py`

**What it does**
- Builds semi-Markov state parameters for the developer model over `OFF/DEV/REV/TEST` by deriving per-developer event timelines from Jira/PR data, estimating transition probabilities, and computing empirical stint PMFs.
- Emits intermediate datasets required to audit the extraction path from raw Jira/PR exports through phase boundaries and developer event sequences.
- Reads the best-fit service-time distributions for DEV/REV/TEST from the Phase 3 distribution summary and writes them to a machine-readable parameter file.

**How it is implemented**
- Loads `etl/output/csv/phase_durations.csv` to obtain per-ticket phase boundaries (`dev_start_ts`, `review_start_ts`, `review_end_ts`, `testing_end_ts`) and skips rows with missing timestamps, recording the reasons in `skipped_phase_rows.csv`.
- Loads `etl/output/csv/github_prs_raw.csv`, extracts Jira keys from PR title/body using the `BOOKKEEPER-<num>` regex, and maps PR assignees to Jira keys for developer attribution.
- Constructs per-developer event stints for DEV/REV/TEST and inserts OFF gaps when there is a positive time gap between consecutive stints; gaps are rounded to a fixed precision (days).
- Computes transition counts between all states and converts them to probabilities using Laplace smoothing (α = 1). Stint PMFs are computed from rounded durations and normalized per state.
- Reads `etl/output/csv/distribution_summary.csv` to capture each phase’s `best_fit` distribution and serialized `parameters`, then writes the values into `service_params.json` without assuming a distribution family.

**How it must be used**
1. Ensure Phase 3 ETL outputs exist:
   - `etl/output/csv/phase_durations.csv`
   - `etl/output/csv/distribution_summary.csv`
2. Run the script from the repository root:
   - `python etl/state_parameters.py`
3. Verify state-parameter outputs under `data/state_parameters/` and review any skipped rows.

**Outputs and contracts**
- Required state-parameter outputs:
  - `data/state_parameters/matrix_P.csv`
  - `data/state_parameters/stint_PMF_OFF.csv`
  - `data/state_parameters/stint_PMF_DEV.csv`
  - `data/state_parameters/stint_PMF_REV.csv`
  - `data/state_parameters/stint_PMF_TEST.csv`
  - `data/state_parameters/service_params.json`
- Intermediate datasets:
  - `data/state_parameters/jira_pr_key_map.csv`
  - `data/state_parameters/developer_ticket_map.csv`
  - `data/state_parameters/developer_events.csv`
  - `data/state_parameters/transition_counts.csv`
  - `data/state_parameters/stint_counts_<STATE>.csv`
  - `data/state_parameters/skipped_phase_rows.csv`
  - `data/state_parameters/skipped_event_rows.csv`
- Limitations:
  - Events with missing or invalid timestamps are skipped and reported.
  - Negative durations or overlaps are discarded and logged in skipped rows.
  - OFF stints are only inferred from observed gaps between sequential events.

---

## ETL Script: `etl/8_stint_PMF.py` — Combined Stint PMF Output

**What it does**
- Computes empirical stint PMFs for `OFF/DEV/REV/TEST` and writes them into a single combined CSV table.
- Uses the same per-developer event construction and stint rounding rules as the state-parameter extraction pipeline.

**How it is implemented**
- Loads `etl/output/csv/phase_durations.csv` to obtain per-ticket phase boundaries and `etl/output/csv/github_prs_raw.csv` to map Jira keys to PR assignees.
- Builds per-developer DEV/REV/TEST events and derives OFF gaps, then rounds stint lengths (days) to a fixed precision before counting durations.
- Normalizes duration counts per state to produce empirical PMFs and validates that probabilities sum to 1.0 within a tolerance.

**How it must be used**
1. Ensure Phase 3 ETL outputs exist:
   - `etl/output/csv/phase_durations.csv`
   - `etl/output/csv/github_prs_raw.csv`
2. Run the script from the repository root:
   - `python etl/8_stint_PMF.py`
3. Review the combined output:
   - `etl/output/csv/stint_PMF.csv`

**Outputs and contracts**
- Required output:
  - `etl/output/csv/stint_PMF.csv` (columns: `state`, `length`, `prob`)
- Limitations:
  - Rows with missing timestamps or non-positive durations are skipped implicitly during stint construction.
  - OFF stints are inferred only when there is a positive gap between consecutive events for a developer.

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

---

## ETL Script: `etl/7_transition_matrix.py`

**What it does**
- Generates the OFF/DEV/REV/TEST transition probability matrix from Jira phase boundary timestamps and GitHub PR assignee mappings.
- Writes the normalized transition matrix to a CSV in `etl/output/csv/`.

**How it is implemented**
- Reads `etl/output/csv/phase_durations.csv` and retains only Jira keys with complete phase timestamps (`dev_start_ts`, `review_start_ts`, `review_end_ts`, `testing_end_ts`).
- Reads `etl/output/csv/github_prs_raw.csv`, extracts Jira keys from PR titles/bodies using the `BOOKKEEPER-<num>` regex, and maps them to PR assignees.
- For each developer, builds ordered DEV/REV/TEST events per ticket, inserts OFF transitions for positive gaps between consecutive events, and counts transitions from OFF to the first observed state.
- Converts transition counts to probabilities using Laplace smoothing (α = 1) and writes the row-normalized matrix to CSV.

**How it must be used**
- **Command:** `python etl/7_transition_matrix.py`
- **Inputs:**
  - `etl/output/csv/phase_durations.csv`
  - `etl/output/csv/github_prs_raw.csv`
- **Outputs:** `etl/output/csv/transition_matrix.csv`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`.
- **Limitations:** Tickets with missing phase timestamps and PRs without assignees or Jira keys are excluded from the transition calculation.
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

## ETL Script: `etl/4_feedback_probabilities_etl.py`

**What it does**
- Extracts PR-level feedback loop probabilities for review → development (changes requested in PR reviews) and testing → development (CI/check failures) using raw GitHub PR export data.
- Filters PRs into a user-supplied time window before computing probabilities.
- Produces a compact CSV summary with counts and probabilities based on the filtered PR set.

**How it is implemented**
- Loads `etl/output/csv/github_prs_raw.csv`.
- Selects a created timestamp column from the available CSV headers (or `--created-col` if supplied), normalizes it to UTC, and filters PRs into the `[start, end)` window supplied on the CLI (defaults to the full data range when omitted).
- Derives review feedback when either:
  - `requested_changes_count` is greater than zero, or
  - `pull_request_review_states` contains `CHANGES_REQUESTED`.
- Derives testing feedback when `check_runs_conclusions` contains a `failure` entry.
- Computes probabilities as feedback-count / total-PR-count within the filtered window (no Jira ticket aggregation).

**How it must be used**
- **Command:** `python etl/4_feedback_probabilities_etl.py [--start <ISO8601>] [--end <ISO8601>] [--created-col <column>]`
- **Inputs:** `etl/output/csv/github_prs_raw.csv`
- **Outputs:**
  - `etl/output/csv/feedback_probabilities.csv`
  - `etl/output/logs/feedback_probabilities.log`
- **Configuration:** Paths are resolved via `path_config.PROJECT_ROOT`.

**Limitations**
- Feedback signals are inferred from PR review/CI data within the supplied window; the script raises an error when required raw columns are missing.

---

## Feedback Probability Extraction (Data-Driven Windowed Signals)

**What it does**
- Computes `FEEDBACK_P_DEV` and `FEEDBACK_P_TEST` from PR data inside a deterministic time window, using raw review and CI signals.

**How it is implemented**
- The ETL script accepts optional `--start` and `--end` timestamps and converts the detected created timestamp column to UTC-naive timestamps before filtering.
- It evaluates feedback at the PR level using `requested_changes_count`/`pull_request_review_states` for review and `check_runs_conclusions` for testing, then computes probabilities as the share of PRs with feedback.

**How it must be used**
- Provide ISO8601 timestamps to define the window when you want to restrict the data slice, and ensure the raw export includes `requested_changes_count` or `pull_request_review_states` plus `check_runs_conclusions`.
- Review the log output to confirm which columns were used and to validate the computed probabilities.

---

## ETL Script: `etl/output/csv/6_extract_initial_dev_count.py`

**What it does**
- Calculates the initial number of developers in each workflow stage (DEV/REV/TEST) plus the number of developers that have an idle (OFF) gap between assignments for simulation initialization.
- Outputs a compact CSV summary for downstream simulation configuration.

**How it is implemented**
- Loads raw Jira issues from `etl/output/csv/jira_issues_raw.csv` and GitHub PR data from `etl/output/csv/github_prs_raw.csv`.
- Extracts Jira keys from PR title/body to associate PRs with Jira issues, then derives phase boundaries:
  - Dev start = Jira created timestamp.
  - Review start = earliest PR created timestamp.
  - Review end = latest PR merged timestamp (fallback to PR closed timestamp).
  - Testing end = Jira resolution timestamp (fallback to review end).
- Collects PR assignees as developers, builds per-developer stage events, then counts how many developers appear in each stage at least once.
- Identifies OFF (idle) developers by detecting gaps between consecutive stage events for each developer, marking them as OFF when a gap exists.

**How it must be used**
- **Command:** `python etl/output/csv/6_extract_initial_dev_count.py`
- **Inputs:**
  - `etl/output/csv/jira_issues_raw.csv`
  - `etl/output/csv/github_prs_raw.csv`
- **Outputs:**
  - `etl/output/csv/initial_dev_count.csv`
- **Limitations:** Requires PR assignees to be present to attribute developers to stages; Jira issues without linked PRs are ignored. OFF counts are derived from gaps between observed stage events.

---

## Update: Initial Developer Counts with OFF (Idle) Gaps

**What it does**
- Extends the initial developer count extraction to include OFF (idle) developers who have at least one gap between consecutive stage events.

**How it is implemented**
- The ETL script now tracks per-developer stage intervals (start/end), uses them to count presence in DEV/REV/TEST, and flags OFF when any time gap is detected between adjacent events.

**How it must be used**
- Run the same command as before; the output CSV now includes an `OFF` row representing idle developers based on observed gaps.

---

## Discrete-Event Simulation Workflow (Finite and Infinite Horizon)

**What it does**
- Implements a discrete-event simulation of the ASF BookKeeper workflow with ticket arrivals, DEV → REVIEW → TESTING stages, and feedback loops back to development.
- Models developer capacity using a semi-Markov developer pool that transitions among OFF/DEV/REV/TEST states to determine concurrent service capacity.
- Produces per-ticket metrics, aggregate performance metrics, and steady-state batch-means summaries with confidence intervals.

**How it is implemented**
- Core DES logic lives in `simulation/core/`:
  - `models.py` defines `Ticket`, `Event`, `SystemState`, and stage/event enums.
  - `engine.py` implements the event queue, time advance, arrival/service-completion handlers, FIFO queues, and routing.
  - `developer_pool.py` implements the semi-Markov developer pool using a transition matrix and stint PMFs, providing time-varying stage capacity.
  - `rng.py` provides stream-specific RNGs for arrivals, services, routing, and developer transitions.
  - `metrics.py` accumulates per-stage throughput/utilization/queue-length time averages and per-ticket time-in-system, plus batch statistics.
  - `outputs.py` writes per-ticket and summary CSVs, and batch means + CI outputs.
- Inputs are loaded from ETL outputs via `inputs.py`, including:
  - Arrival rate (`etl/output/csv/arrival_rate_jira_issues.csv`).
  - Feedback probabilities (`etl/output/csv/feedback_probabilities.csv`).
  - Developer transition matrix (`etl/output/csv/transition_matrix.csv`).
  - Stint PMFs (`etl/output/csv/stint_PMF.csv`).
  - Service time distributions (`data/state_parameters/service_params.json`, with fallback to `etl/output/csv/distribution_summary.csv`).
  - Developer count inferred from `etl/output/csv/initial_dev_count.csv` when available, or from `data/state_parameters/developer_events.csv`.
- Randomness is deterministic when the seed is fixed; each stochastic input uses its own RNG stream.

**How to run it**
1. Ensure ETL outputs listed above exist (run ETL scripts in prior sections).
2. Run the finite-horizon simulation (365 days by default):
   - `python simulation/finite_horizon/run_simulation.py --seed 12345 --horizon 365`
   - Or make the script executable and run it directly: `chmod +x simulation/finite_horizon/run_simulation.py && simulation/finite_horizon/run_simulation.py --seed 12345 --horizon 365`
   - Outputs:
     - `simulation/finite_horizon/output/tickets.csv`
     - `simulation/finite_horizon/output/summary.csv`
3. Run the infinite-horizon batch-means simulation (3,650 days, 10 batches by default):
   - `python simulation/infinite_horizon/run_simulation.py --seed 12345 --total-time 3650 --batches 10`
   - Or make the script executable and run it directly: `chmod +x simulation/infinite_horizon/run_simulation.py && simulation/infinite_horizon/run_simulation.py --seed 12345 --total-time 3650 --batches 10`
   - Outputs:
     - `simulation/infinite_horizon/output/summary_batch_means.csv`
     - `simulation/infinite_horizon/output/summary_ci.csv`

**Limitations and assumptions**
- Requires precomputed ETL outputs; missing files will prevent the simulation from running.
- Service-time distributions are assumed to be parametric fits (lognormal/Weibull/exponential) derived by ETL scripts.
- Developer capacity changes are driven by the semi-Markov model; if capacity drops below busy servers, service continues but no new work starts until capacity is available.
- Confidence intervals use a normal approximation over batch means (no external stats dependency).

---

## Simulation Input Calibration and Diagnostics (Service-Time Validation)

**What it does**
- Adds a calibration step that compares ETL phase-duration samples to the fitted service-time distributions and applies a conservative cap at the empirical 99th percentile to prevent unrealistic heavy-tail stalls.
- Converts service-time samples from hours to days when phase-duration inputs are expressed in hours, keeping the simulation time unit consistent with the arrival rate (per day).
- Writes a diagnostics CSV that reports the empirical statistics and the configured cap/scale values used during the run.

**How it is implemented**
- The runners load `etl/output/csv/phase_durations.csv` and compute per-stage empirical statistics (mean, p95, p99, max) from the `*_duration_hours` columns.
- A service-time scale factor of `1/24` is applied when the input columns are in hours, ensuring all service times are in days.
- The service-time cap is set to the empirical p99 (in days) for each stage; sampled service times above the cap are clipped.
- Diagnostics are written to `service_time_diagnostics.csv` in each output folder, capturing the fitted distribution, parameters, scale, and empirical stats.

**How it must be used**
1. Ensure `etl/output/csv/phase_durations.csv` is available (produced by the phase-duration ETL scripts).
2. Run either simulation script as usual.
3. Review diagnostics output:
   - `simulation/finite_horizon/output/service_time_diagnostics.csv`
   - `simulation/infinite_horizon/output/service_time_diagnostics.csv`

**Limitations and assumptions**
- The cap uses empirical percentiles and will clip rare extreme samples; this is intended to keep simulated timelines aligned with observed ETL data.
- If `phase_durations.csv` is missing or empty, the simulation runs without caps or scaling and diagnostics will reflect empty empirical stats.

---

## Empirical Service-Time Sampling (Trace-Driven Option)

**What it does**
- Uses the empirical phase-duration samples from `etl/output/csv/phase_durations.csv` directly (trace-driven sampling) to generate service times, aligning simulation service times with observed data.
- Falls back to the fitted parametric distributions when no empirical samples are available for a stage.

**How it is implemented**
- The runners load per-stage duration samples from the `*_duration_hours` columns and convert them to days (scale `1/24`).
- Service times are drawn uniformly at random from the empirical sample list for each stage when available; caps remain in place as a safety check.
- The diagnostics CSV includes a `sampling_mode` column and the `sample_count` to show whether a stage used empirical or parametric sampling.

**How it must be used**
1. Ensure `etl/output/csv/phase_durations.csv` exists and contains non-zero samples.
2. Run the simulations as usual.
3. Inspect the diagnostics:
   - `simulation/finite_horizon/output/service_time_diagnostics.csv`
   - `simulation/infinite_horizon/output/service_time_diagnostics.csv`

**Limitations and assumptions**
- Empirical sampling mirrors the historical dataset; it does not extrapolate beyond observed durations.
