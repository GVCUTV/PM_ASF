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
