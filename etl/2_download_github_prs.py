# v8
# file: etl/2_download_github_prs.py
"""
GitHub PR downloader for apache/bookkeeper with:
- Concurrent page fetching for the PR list (state=all).
- Concurrent per-PR detail fetching (reviews, check-runs, combined statuses).
- Shared HTTP session with pooled connections + QPS throttle.
- ETag cache (disk-persisted) for conditional GETs; 304s reuse cached bodies.
- Incremental mode: skip detail calls for PRs whose `updated_at` didn't change since last run.
- **Multi-token rotation**: on rate-limit (403/Remaining=0) auto-switch to next token; sleep only if all tokens are exhausted.
- Robust 403 handling using X-RateLimit-Remaining/Reset and Retry-After when present.
- Tokens loaded from etl/env/github.env (GITHUB_TOKENS=… or multiple GITHUB_TOKEN=…) and/or env var GITHUB_TOKENS.
- Exhaustive logging to stdout and output/logs/download_github_prs.log.

Repo: https://github.com/GVCUTV/BK_ASF.git
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode

from path_config import PROJECT_ROOT  # CWD-independent

# --------------------------- Constants & Paths --------------------------- #

LOG_DIR = path.join(PROJECT_ROOT, "output", "logs")
OUT_CSV = path.join(PROJECT_ROOT, "etl", "output", "csv", "github_prs_raw.csv")
ENV_FILE = path.join(PROJECT_ROOT, "etl", "env", "github_tokens.env")
CACHE_FILE = path.join(PROJECT_ROOT, "etl", "cache", "github_http_cache.json")

OWNER = "apache"
REPO = "bookkeeper"
PER_PAGE = 100

MAX_WORKERS = int(os.getenv("GITHUB_LIST_WORKERS", "10"))        # PR list pagination workers
DETAIL_WORKERS = int(os.getenv("GITHUB_DETAIL_WORKERS", "12"))   # per-PR details workers
RETRIES = 3
TIMEOUT = 30

# QPS throttle to avoid secondary limits (gentle by default)
GITHUB_QPS = float(os.getenv("GITHUB_QPS", "6.0"))

# Connection pool sizes for the shared Session
POOL_CONNECTIONS = int(os.getenv("GITHUB_POOL_CONNS", str(max(32, DETAIL_WORKERS * 2))))
POOL_MAXSIZE = int(os.getenv("GITHUB_POOL_MAXSIZE", str(max(64, DETAIL_WORKERS * 4))))

# Incremental mode: reuse previous CSV rows if PR `updated_at` is unchanged
INCREMENTAL = os.getenv("GITHUB_INCREMENTAL", "1") not in ("0", "false", "False")

# --------------------------- Logging --------------------------- #

def _safe_mkdirs(d: str):
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        if not path.isdir(d):
            raise

def _setup_logging():
    _safe_mkdirs(LOG_DIR)
    log_path = path.join(LOG_DIR, "download_github_prs.log")

    root = logging.getLogger()
    root.handlers[:] = []
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)

    logging.info("Logger ready. Logfile: %s", log_path)
    return log_path

# --------------------------- Tokens --------------------------- #

def _read_tokens_from_envfile(p: str) -> List[str]:
    tokens: List[str] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
    except IOError:
        logging.warning("Token file not found: %s", p)
        text = ""

    def _add_candidates(s: str):
        # Accept both GITHUB_TOKENS and GITHUB_TOKEN lines; allow comma/space separated.
        for line in s.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.search(r"(GITHUB_TOKENS?|PAT)\s*=\s*(.+)$", line, flags=re.IGNORECASE)
            if m:
                rhs = m.group(2).strip().strip("'\"")
                for tok in re.split(r"[, \t]+", rhs):
                    tok = tok.strip()
                    if tok:
                        tokens.append(tok)

    _add_candidates(text)

    env_multi = os.getenv("GITHUB_TOKENS", "")
    if env_multi:
        logging.info("Detected GITHUB_TOKENS env var.")
        _add_candidates(f"GITHUB_TOKENS={env_multi}")

    # Deduplicate keeping order
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

class TokenPool:
    """
    Thread-safe pool that rotates among multiple tokens.
    - Updates per-token remaining/reset from response headers.
    - On primary rate limit (remaining=0 / 403 "rate limit"), switches to a token with quota.
    - If all tokens are exhausted, sleeps until earliest reset.
    """
    def __init__(self, tokens: List[Optional[str]]):
        self.tokens = tokens or [None]
        self.n = len(self.tokens)
        self.idx = 0
        self.meta = [{"remaining": None, "reset": None} for _ in self.tokens]
        self.lock = threading.Lock()

        masked = [self._mask(t) for t in self.tokens]
        logging.info("TokenPool initialized with %d token(s): %s", self.n, masked)

    @staticmethod
    def _mask(tok: Optional[str]) -> str:
        if tok is None:
            return "<anonymous>"
        if len(tok) <= 8:
            return tok[:2] + "…" + tok[-2:]
        return tok[:4] + "…" + tok[-4:]

    def current(self) -> Optional[str]:
        with self.lock:
            return self.tokens[self.idx]

    def header(self) -> dict:
        tok = self.current()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def update_from_headers(self, headers: dict):
        """Store Remaining/Reset for current token if headers present."""
        with self.lock:
            try:
                rem = headers.get("X-RateLimit-Remaining")
                rst = headers.get("X-RateLimit-Reset")
                if rem is not None:
                    self.meta[self.idx]["remaining"] = int(rem)
                if rst is not None:
                    self.meta[self.idx]["reset"] = int(rst)
            except Exception:
                pass  # be resilient

    def on_rate_limited(self, headers: dict, body_text: str) -> str:
        """
        Handle 403/limit. Returns:
          - "switched" if we moved to a different token,
          - "slept"    if we slept until reset (all tokens exhausted).
        """
        # First, update current token's meta from headers
        self.update_from_headers(headers)

        msg = (body_text or "").lower()
        limited = True if "rate limit" in msg or "abuse detection" in msg else False
        with self.lock:
            cur = self.idx
            now = int(time.time())

            # Try to find a token with remaining>0 or reset already passed (unknown remaining treated as usable)
            for hop in range(1, self.n + 1):
                cand = (cur + hop) % self.n
                m = self.meta[cand]
                usable = (
                    self.tokens[cand] is not None and
                    (m["remaining"] is None or m["remaining"] > 0 or (m["reset"] is not None and m["reset"] <= now))
                )
                if usable:
                    self.idx = cand
                    logging.warning("Rate limited on token %s; switching to token %s.",
                                    self._mask(self.tokens[cur]), self._mask(self.tokens[cand]))
                    return "switched"

            # No usable token -> compute earliest reset and sleep
            resets = [m["reset"] for m in self.meta if m["reset"] is not None]
            earliest = min(resets) if resets else None

        # Sleep OUTSIDE the lock
        if earliest is None:
            # Fallback: sleep a short period
            logging.warning("All tokens appear limited but no reset header available. Sleeping 60s.")
            time.sleep(60)
            return "slept"
        else:
            wait = max(0, earliest - int(time.time()) + 1)
            logging.warning("All tokens limited. Sleeping %ss until earliest reset (%s).", wait, earliest)
            if wait > 0:
                time.sleep(wait)
            return "slept"

# Build the token pool
_TOKENS = _read_tokens_from_envfile(ENV_FILE)
if not _TOKENS:
    logging.warning("No tokens found; operating anonymously (very low limits).")
token_pool = TokenPool(_TOKENS if _TOKENS else [None])

def _has_auth() -> bool:
    return token_pool.current() is not None

# --------------------------- Shared Session, Throttle, Cache --------------------------- #

_SESSION = requests.Session()
_ADAPTER = HTTPAdapter(pool_connections=POOL_CONNECTIONS, pool_maxsize=POOL_MAXSIZE, max_retries=0)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

_RATE_LOCK = threading.Lock()
_LAST_REQ_TS = 0.0

def _throttle():
    """Global QPS throttle to avoid tripping secondary limits."""
    global _LAST_REQ_TS
    with _RATE_LOCK:
        min_interval = 1.0 / max(GITHUB_QPS, 0.1)
        now = time.monotonic()
        wait = (_LAST_REQ_TS + min_interval) - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _LAST_REQ_TS = now

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}

def _cache_key(url: str, params: dict | None, preview: bool) -> str:
    qp = urlencode(sorted((params or {}).items()))
    raw = f"{url}?{qp}|preview={1 if preview else 0}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _load_cache():
    _safe_mkdirs(path.dirname(CACHE_FILE))
    if path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _CACHE.update(data)
                logging.info("Loaded ETag cache entries: %d", len(_CACHE))
        except Exception as e:
            logging.warning("Failed to load cache %s: %s", CACHE_FILE, e)

def _save_cache():
    try:
        with _CACHE_LOCK:
            tmp = CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_CACHE, f)
            os.replace(tmp, CACHE_FILE)
        logging.info("Saved ETag cache entries: %d -> %s", len(_CACHE), CACHE_FILE)
    except Exception as e:
        logging.warning("Failed to persist cache: %s", e)

atexit.register(_save_cache)

# --------------------------- HTTP helpers --------------------------- #

def _headers(preview: bool = False, etag: str | None = None) -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "BK_ASF-ETL",
        "Connection": "keep-alive",
    }
    # add Authorization if we have a token
    h.update(token_pool.header())
    if preview:
        h["Accept"] = "application/vnd.github.antiope-preview+json"
    if etag:
        h["If-None-Match"] = etag
    return h

def _req_get(url, params=None, preview=False, return_response=False):
    """
    GET with retries, QPS throttle, conditional ETag headers, token rotation, and reset-aware sleep.
    Returns r.json() (or cached body on 304) or raw response if return_response=True.
    """
    key = _cache_key(url, params, preview)
    with _CACHE_LOCK:
        cached = _CACHE.get(key, {})
        etag = cached.get("etag")

    for attempt in range(RETRIES + 1):
        _throttle()
        try:
            r = _SESSION.get(url, headers=_headers(preview=preview, etag=etag), params=params, timeout=TIMEOUT)
            status = r.status_code

            # Update token metadata from headers
            token_pool.update_from_headers(r.headers)

            # 304 Not Modified -> use cached body (doesn't spend primary limit)
            if status == 304:
                with _CACHE_LOCK:
                    body = cached.get("body")
                if body is not None:
                    logging.info("HTTP 304 cache-hit for %s", url)
                    return r if return_response else body
                # Shouldn't happen; retry without If-None-Match
                etag = None
                logging.warning("304 received but cache empty. Retrying unconditionally.")
                time.sleep(1.0)
                continue

            if status == 200:
                if return_response:
                    return r
                body = r.json()
                # store ETag + body for next time
                with _CACHE_LOCK:
                    _CACHE[key] = {
                        "etag": r.headers.get("ETag"),
                        "body": body,
                        "ts": int(time.time()),
                    }
                return body

            # Primary/secondary rate limit -> 403 with rate headers
            if status == 403 and ("rate limit" in r.text.lower() or r.headers.get("X-RateLimit-Remaining") == "0"):
                action = token_pool.on_rate_limited(r.headers, r.text)
                if action in ("switched", "slept"):
                    # Retry immediately with the (possibly) new token
                    continue

            # Transient server/network errors
            if status in (429, 502, 503, 504):
                backoff = 2 * (attempt + 1)
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        backoff = max(backoff, int(ra))
                    except Exception:
                        pass
                logging.warning("Transient error %s on %s (attempt %d). Sleeping %ss.",
                                status, url, attempt + 1, backoff)
                time.sleep(backoff)
                continue

            # Other errors -> log and retry a bit
            logging.warning("GET %s -> %s | %s", url, status, r.text[:200])
            time.sleep(2 * (attempt + 1))
        except requests.RequestException as e:
            logging.warning("Exception GET %s: %s (attempt %d)", url, e, attempt + 1)
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("GET failed after retries: %s" % url)

# --------------------------- Pagination helpers --------------------------- #

def _discover_last_page(owner, repo, per_page=PER_PAGE):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "all", "per_page": per_page, "page": 1}
    r = _req_get(url, params=params, return_response=True)
    link = r.headers.get("Link", "")
    if link:
        m = re.search(r'[?&]page=(\d+)[^>]*>; rel="last"', link)
        if m:
            last = int(m.group(1))
            logging.info("Discovered last page via Link header: %d", last)
            return last
    data = r.json()
    n = len(data) if isinstance(data, list) else 0
    last = 1 if n < per_page else 2
    logging.info("No Link header. First page size=%d -> tentative last=%d", n, last)
    return last

def _fetch_pr_page(owner, repo, page, per_page=PER_PAGE):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {"state": "all", "per_page": per_page, "page": page}
    data = _req_get(url, params=params)
    if isinstance(data, list):
        return page, data
    logging.warning("Unexpected PR page payload (not list) at page=%d", page)
    return page, []

def _list_all_prs_concurrent(owner, repo, per_page=PER_PAGE, max_workers=MAX_WORKERS):
    last_page = _discover_last_page(owner, repo, per_page)
    pages = list(range(1, last_page + 1))
    logging.info("Fetching %d pages concurrently with %d workers…", len(pages), max_workers)

    prs_by_page = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_pr_page, owner, repo, p, per_page): p for p in pages}
        for fut in as_completed(futures):
            page, data = fut.result()
            prs_by_page[page] = data
            logging.info("Page %d fetched: %d PRs", page, len(data))

    all_pages = sorted(prs_by_page.keys())
    while all_pages and len(prs_by_page[all_pages[-1]]) == 0:
        logging.info("Trimming empty trailing page %d", all_pages[-1])
        all_pages.pop()

    flattened = []
    for p in all_pages:
        flattened.extend(prs_by_page[p])

    logging.info("Total PRs fetched: %d", len(flattened))
    return flattened

# --------------------------- Detail helpers --------------------------- #

def _list_reviews(owner, repo, pr_number):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    data = _req_get(url, preview=False)
    if isinstance(data, list):
        return data
    logging.warning("Unexpected reviews payload for PR#%s (type=%s)", pr_number, type(data).__name__)
    return []

def _list_check_runs(owner, repo, sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/check-runs"
    data = _req_get(url, preview=True)
    if isinstance(data, dict):
        runs = data.get("check_runs") or []
        if isinstance(runs, list):
            return runs
    elif isinstance(data, list):
        return data
    elif isinstance(data, (str, bytes)):
        logging.warning("Checks payload is a %s string for sha=%s; skipping.", type(data).__name__, sha)
    else:
        logging.warning("Unexpected checks payload type=%s for sha=%s", type(data).__name__, sha)
    return []

def _combined_status(owner, repo, sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}/status"
    data = _req_get(url, preview=False)
    if isinstance(data, dict):
        statuses = data.get("statuses") or []
        if isinstance(statuses, list):
            return statuses
    elif isinstance(data, list):
        return data
    elif isinstance(data, (str, bytes)):
        logging.warning("Status payload is a %s string for sha=%s; skipping.", type(data).__name__, sha)
    else:
        logging.warning("Unexpected status payload type=%s for sha=%s", type(data).__name__, sha)
    return []

def _derive_review_signals(reviews):
    states = []
    try:
        for x in reviews:
            if isinstance(x, dict):
                states.append(str(x.get("state", "")).upper())
    except Exception as e:
        logging.warning("Review states parse error: %s", e)
    requested_changes = sum(1 for s in states if s == "CHANGES_REQUESTED")
    return {
        "reviews_count": len(reviews) if isinstance(reviews, list) else 0,
        "requested_changes_count": requested_changes,
        "pull_request_review_states": json.dumps(states),
    }

def _derive_checks(runs):
    conclusions = []
    try:
        for x in runs if isinstance(runs, list) else []:
            if isinstance(x, dict):
                conclusions.append(str(x.get("conclusion") or "").lower())
    except Exception as e:
        logging.warning("Checks parse error: %s", e)
    return {"check_runs_conclusions": json.dumps(conclusions)}

def _derive_statuses(statuses):
    states = []
    try:
        for x in statuses if isinstance(statuses, list) else []:
            if isinstance(x, dict):
                states.append(str(x.get("state") or "").lower())
    except Exception as e:
        logging.warning("Statuses parse error: %s", e)
    return {"combined_status_states": json.dumps(states)}

# --------------------------- Incremental support --------------------------- #

def _load_previous_index():
    """
    Load previous CSV (if any) into:
      - prev_rows: map PR number -> dict row
      - prev_updated: map PR number -> updated_at string
    """
    prev_rows, prev_updated = {}, {}
    if path.isfile(OUT_CSV):
        try:
            prev = pd.read_csv(OUT_CSV, dtype={"number": "Int64"})
            for _, row in prev.iterrows():
                num = int(row["number"])
                prev_rows[num] = dict(row)
                prev_updated[num] = str(row.get("updated_at") or "")
            logging.info("Loaded previous CSV with %d rows for incremental reuse.", len(prev_rows))
        except Exception as e:
            logging.warning("Failed to load previous CSV: %s", e)
    return prev_rows, prev_updated

# --------------------------- Per-PR processing (parallelizable) --------------------------- #

def _process_one_pr(pr):
    t0 = time.time()
    number = pr.get("number")
    head = pr.get("head") or {}
    head_sha = head.get("sha")

    reviews = _list_reviews(OWNER, REPO, number)
    rev_sig = _derive_review_signals(reviews)

    chk_sig = {"check_runs_conclusions": json.dumps([])}
    if head_sha:
        runs = _list_check_runs(OWNER, REPO, head_sha)
        chk_sig = _derive_checks(runs)

    st_sig = {"combined_status_states": json.dumps([])}
    if head_sha:
        statuses = _combined_status(OWNER, REPO, head_sha)
        st_sig = _derive_statuses(statuses)

    base = {
        "number": number,
        "html_url": pr.get("html_url"),
        "state": pr.get("state"),
        "title": pr.get("title"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "closed_at": pr.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "merge_commit_sha": pr.get("merge_commit_sha"),
        "user.login": (pr.get("user") or {}).get("login"),
        "assignee.login": (pr.get("assignee") or {}).get("login"),
        "requested_reviewers": json.dumps([(u or {}).get("login") for u in (pr.get("requested_reviewers") or [])]),
        "head.ref": head.get("ref"),
        "head.sha": head_sha,
        "base.ref": (pr.get("base") or {}).get("ref"),
    }
    base.update(rev_sig)
    base.update(chk_sig)
    base.update(st_sig)

    dt = time.time() - t0
    logging.info("Processed PR #%s in %.2fs | reviews=%s, changes_requested=%s, head_sha=%s",
                 str(number), dt, str(base.get("reviews_count")),
                 str(base.get("requested_changes_count")), head_sha if head_sha else "-")
    return base

# --------------------------- Main --------------------------- #

def main():
    _setup_logging()
    _load_cache()
    logging.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logging.info("OUT_CSV     : %s", OUT_CSV)
    logging.info("Workers     : list=%d, details=%d, pool_conns=%d, pool_max=%d, qps=%.2f",
                 MAX_WORKERS, DETAIL_WORKERS, POOL_CONNECTIONS, POOL_MAXSIZE, GITHUB_QPS)
    if not _has_auth():
        logging.warning("No auth token found. You will be limited by GitHub's anonymous rate limits.")

    # 0) Load previous CSV index for incremental mode
    prev_rows, prev_updated = _load_previous_index()

    # 1) Fetch PR list (concurrent across pages)
    prs = _list_all_prs_concurrent(OWNER, REPO, per_page=PER_PAGE, max_workers=MAX_WORKERS)
    if not prs:
        logging.warning("No PRs found.")
        _safe_mkdirs(path.dirname(OUT_CSV))
        pd.DataFrame([]).to_csv(OUT_CSV, index=False)
        return

    # 1.5) Partition PRs into: reused (unchanged) vs to_process
    rows = []
    to_process = []
    if INCREMENTAL and prev_rows:
        for pr in prs:
            num = pr.get("number")
            upd = str(pr.get("updated_at") or "")
            if num in prev_updated and prev_updated[num] == upd:
                # reuse previous row; avoids detail calls entirely
                rows.append(prev_rows[num])
            else:
                to_process.append(pr)
        logging.info("Incremental: reused=%d, to_fetch=%d (of %d total)",
                     len(rows), len(to_process), len(prs))
    else:
        to_process = prs

    # 2) For each PR to process, fetch details in parallel
    t_loop0 = time.time()
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as ex:
        futures = [ex.submit(_process_one_pr, pr) for pr in to_process]
        processed = 0
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - t_loop0
                    logging.info("Parallel details progress: %d/%d (%.1f%%) in %.1fs",
                                 processed, len(to_process),
                                 100.0 * processed / max(1, len(to_process)), elapsed)
            except Exception as e:
                logging.warning("Error processing PR item in worker: %s", e)

    # 3) Save CSV
    df = pd.DataFrame(rows)
    _safe_mkdirs(path.dirname(OUT_CSV))
    df.to_csv(OUT_CSV, index=False)
    logging.info("Saved %d PRs in %s", len(df), OUT_CSV)

    # 4) Quick stats
    for c in ["reviews_count", "requested_changes_count"]:
        if c in df.columns:
            logging.info("Stats %s: non-null=%d mean=%.4f max=%s",
                         c, int(df[c].notna().sum()),
                         float(pd.to_numeric(df[c], errors='coerce').mean()),
                         str(pd.to_numeric(df[c], errors='coerce').max()))
    if "check_runs_conclusions" in df.columns and len(df) > 0:
        logging.info("Sample check_runs_conclusions: %s", df["check_runs_conclusions"].iloc[0])
    if "combined_status_states" in df.columns and len(df) > 0:
        logging.info("Sample combined_status_states: %s", df["combined_status_states"].iloc[0])

if __name__ == "__main__":
    main()
