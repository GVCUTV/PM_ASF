# v1
# file: 10_phase_duration_distribution_etl.py

"""
Compute per-ticket phase durations (dev/review/testing) and distribution summaries
from raw Jira/GitHub exports, aligned to source_of_truth boundary rules.
"""

import csv
import math
import os
import re
import statistics
import struct
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from path_config import PROJECT_ROOT

JIRA_TICKETS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_tickets_raw.csv"
JIRA_ISSUES_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "jira_issues_raw.csv"
PRS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "github_prs_raw.csv"

PHASE_DURATIONS_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "phase_durations.csv"
DISTRIBUTION_SUMMARY_CSV = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "distribution_summary.csv"
DISTRIBUTION_SUMMARY_MD = Path(PROJECT_ROOT) / "etl" / "output" / "csv" / "distribution_summary.md"
PLOTS_DIR = Path(PROJECT_ROOT) / "etl" / "output" / "plots"

JIRA_KEY_REGEX = re.compile(r"BOOKKEEPER-\d+", re.IGNORECASE)


@dataclass
class TicketPhase:
    key: str
    dev_start: datetime | None
    review_start: datetime | None
    review_end: datetime | None
    testing_end: datetime | None
    dev_duration_hours: float | None
    review_duration_hours: float | None
    testing_duration_hours: float | None
    exception_reason: str


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
    ):
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_jira_keys(*values: str | None) -> set[str]:
    matches: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in JIRA_KEY_REGEX.findall(str(value)):
            matches.add(match.upper())
    return matches


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (pct / 100)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_values[int(k)]
    weight = k - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def log_likelihood_exponential(samples: list[float]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    mean_val = statistics.mean(samples)
    if mean_val <= 0:
        return None, None
    rate = 1.0 / mean_val
    loglik = len(samples) * math.log(rate) - rate * sum(samples)
    return loglik, rate


def log_likelihood_lognormal(samples: list[float]) -> tuple[float | None, tuple[float, float] | None]:
    positives = [s for s in samples if s > 0]
    if not positives:
        return None, None
    log_samples = [math.log(s) for s in positives]
    mu = statistics.mean(log_samples)
    sigma = statistics.pstdev(log_samples)
    if sigma == 0:
        return None, None
    loglik = 0.0
    for s, ln_s in zip(positives, log_samples):
        loglik += -math.log(s * sigma * math.sqrt(2 * math.pi)) - ((ln_s - mu) ** 2) / (2 * sigma ** 2)
    return loglik, (mu, sigma)


def estimate_weibull_params(samples: list[float]) -> tuple[float, float] | None:
    positives = [s for s in samples if s > 0]
    if len(positives) < 2:
        return None
    log_samples = [math.log(s) for s in positives]
    mean_log = statistics.mean(log_samples)
    std_log = statistics.pstdev(log_samples)
    if std_log == 0:
        return None
    shape = max(0.1, 1.2 / std_log)
    for _ in range(25):
        sum_xk = sum(s ** shape for s in positives)
        sum_xk_log = sum((s ** shape) * math.log(s) for s in positives)
        if sum_xk == 0:
            break
        shape_next = 1.0 / ((sum_xk_log / sum_xk) - mean_log)
        if abs(shape_next - shape) < 1e-6:
            shape = shape_next
            break
        shape = shape_next
    scale = (sum(s ** shape for s in positives) / len(positives)) ** (1.0 / shape)
    return shape, scale


def log_likelihood_weibull(samples: list[float]) -> tuple[float | None, tuple[float, float] | None]:
    params = estimate_weibull_params(samples)
    if params is None:
        return None, None
    shape, scale = params
    positives = [s for s in samples if s > 0]
    if not positives:
        return None, None
    loglik = 0.0
    for s in positives:
        loglik += (
            math.log(shape)
            - shape * math.log(scale)
            + (shape - 1) * math.log(s)
            - (s / scale) ** shape
        )
    return loglik, params


def load_jira_rows() -> tuple[list[dict[str, str]], list[str], Path]:
    jira_path = JIRA_TICKETS_CSV if JIRA_TICKETS_CSV.exists() else JIRA_ISSUES_CSV
    if not jira_path.exists():
        raise FileNotFoundError("Missing Jira CSV export.")
    with jira_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames or []
    return rows, header, jira_path


def load_pr_rows() -> list[dict[str, str]]:
    with PRS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_pr_index(pr_rows: list[dict[str, str]]) -> dict[str, dict[str, datetime | None]]:
    pr_index: dict[str, dict[str, datetime | None]] = {}
    for row in pr_rows:
        keys = extract_jira_keys(row.get("title"), row.get("body"))
        if not keys:
            continue
        created = parse_timestamp(row.get("created_at"))
        merged = parse_timestamp(row.get("merged_at"))
        closed = parse_timestamp(row.get("closed_at"))
        for key in keys:
            stats = pr_index.setdefault(
                key,
                {"first_created": None, "last_merged": None, "last_closed": None},
            )
            if created and (stats["first_created"] is None or created < stats["first_created"]):
                stats["first_created"] = created
            if merged and (stats["last_merged"] is None or merged > stats["last_merged"]):
                stats["last_merged"] = merged
            if closed and (stats["last_closed"] is None or closed > stats["last_closed"]):
                stats["last_closed"] = closed
    return pr_index


def compute_phase_durations(
    jira_rows: list[dict[str, str]],
    pr_index: dict[str, dict[str, datetime | None]],
    key_field: str,
    created_field: str,
    resolution_field: str,
) -> list[TicketPhase]:
    phases: list[TicketPhase] = []
    for row in jira_rows:
        key = (row.get(key_field) or "").strip().upper()
        if not key:
            continue
        dev_start = parse_timestamp(row.get(created_field))
        resolution = parse_timestamp(row.get(resolution_field))

        pr_stats = pr_index.get(key, {})
        review_start = pr_stats.get("first_created") if pr_stats else None
        review_end = pr_stats.get("last_merged") or pr_stats.get("last_closed") if pr_stats else None
        testing_end = resolution or review_end

        exception_reasons = []
        if dev_start is None:
            exception_reasons.append("missing_dev_start")
        if review_start is None:
            exception_reasons.append("missing_review_start")
        if review_end is None:
            exception_reasons.append("missing_review_end")
        if testing_end is None:
            exception_reasons.append("missing_testing_end")

        # Phase boundary derivation is aligned to source_of_truth fallback rules.
        dev_duration = None
        if dev_start and review_start:
            dev_duration = (review_start - dev_start).total_seconds() / 3600.0
            if dev_duration < 0:
                dev_duration = None
                exception_reasons.append("negative_dev_duration")

        review_duration = None
        if review_start and review_end:
            review_duration = (review_end - review_start).total_seconds() / 3600.0
            if review_duration < 0:
                review_duration = None
                exception_reasons.append("negative_review_duration")

        testing_duration = None
        if review_end and testing_end:
            testing_duration = (testing_end - review_end).total_seconds() / 3600.0
            if testing_duration < 0:
                testing_duration = None
                exception_reasons.append("negative_testing_duration")

        phases.append(
            TicketPhase(
                key=key,
                dev_start=dev_start,
                review_start=review_start,
                review_end=review_end,
                testing_end=testing_end,
                dev_duration_hours=dev_duration,
                review_duration_hours=review_duration,
                testing_duration_hours=testing_duration,
                exception_reason=";".join(exception_reasons),
            )
        )
    return phases


def compute_summary(samples: list[float]) -> dict[str, float | int | str | None]:
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "variance": None,
            "stdev": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    sorted_vals = sorted(samples)
    variance = statistics.variance(sorted_vals) if len(sorted_vals) > 1 else None
    stdev = statistics.stdev(sorted_vals) if len(sorted_vals) > 1 else None
    return {
        "count": len(sorted_vals),
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "variance": variance,
        "stdev": stdev,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p10": percentile(sorted_vals, 10),
        "p25": percentile(sorted_vals, 25),
        "p75": percentile(sorted_vals, 75),
        "p90": percentile(sorted_vals, 90),
    }


def select_best_fit(samples: list[float]) -> tuple[str | None, dict[str, float | None]]:
    logliks = {}
    exp_loglik, _ = log_likelihood_exponential(samples)
    if exp_loglik is not None:
        logliks["exponential"] = exp_loglik
    lognorm_loglik, _ = log_likelihood_lognormal(samples)
    if lognorm_loglik is not None:
        logliks["lognormal"] = lognorm_loglik
    weibull_loglik, _ = log_likelihood_weibull(samples)
    if weibull_loglik is not None:
        logliks["weibull"] = weibull_loglik
    if not logliks:
        return None, {"exponential": None, "lognormal": None, "weibull": None}
    best = max(logliks.items(), key=lambda item: item[1])[0]
    return best, {
        "exponential": logliks.get("exponential"),
        "lognormal": logliks.get("lognormal"),
        "weibull": logliks.get("weibull"),
    }


def write_phase_durations(phases: list[TicketPhase]) -> None:
    with PHASE_DURATIONS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "jira_key",
                "dev_start_ts",
                "review_start_ts",
                "review_end_ts",
                "testing_end_ts",
                "dev_duration_hours",
                "review_duration_hours",
                "testing_duration_hours",
                "exception_reason",
            ]
        )
        for phase in phases:
            writer.writerow(
                [
                    phase.key,
                    format_timestamp(phase.dev_start),
                    format_timestamp(phase.review_start),
                    format_timestamp(phase.review_end),
                    format_timestamp(phase.testing_end),
                    f"{phase.dev_duration_hours:.4f}" if phase.dev_duration_hours is not None else "",
                    f"{phase.review_duration_hours:.4f}" if phase.review_duration_hours is not None else "",
                    f"{phase.testing_duration_hours:.4f}" if phase.testing_duration_hours is not None else "",
                    phase.exception_reason,
                ]
            )


def write_distribution_summary(
    phase_samples: dict[str, list[float]],
    best_fits: dict[str, str | None],
) -> None:
    with DISTRIBUTION_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase",
                "count",
                "mean",
                "median",
                "variance",
                "stdev",
                "min",
                "max",
                "p10",
                "p25",
                "p75",
                "p90",
                "best_fit",
            ]
        )
        for phase, samples in phase_samples.items():
            summary = compute_summary(samples)
            writer.writerow(
                [
                    phase,
                    summary["count"],
                    summary["mean"],
                    summary["median"],
                    summary["variance"],
                    summary["stdev"],
                    summary["min"],
                    summary["max"],
                    summary["p10"],
                    summary["p25"],
                    summary["p75"],
                    summary["p90"],
                    best_fits.get(phase),
                ]
            )


def write_distribution_summary_md(
    phase_samples: dict[str, list[float]],
    best_fits: dict[str, str | None],
    exception_keys: list[str],
    exception_sample: list[str],
    jira_path: Path,
    has_transitions: bool,
) -> None:
    lines = [
        "# Phase Duration Distribution Summary",
        "",
        f"Source Jira file: `{jira_path}`.",
        f"Source PR file: `{PRS_CSV}`.",
        "",
        "## Phase boundary assumptions",
        "- Dev start: Jira issue creation timestamp.",
        "- Review start: earliest PR creation timestamp linked to the Jira key.",
        "- Review end: latest PR merge timestamp (fallback to PR close if merge missing).",
        "- Testing end: Jira resolution timestamp (fallback to review end if missing).",
        "",
        "## Distribution summary",
    ]
    for phase, samples in phase_samples.items():
        lines.append(f"- **{phase}**: {len(samples)} samples, best fit `{best_fits.get(phase)}`.")
    lines.extend(
        [
            "",
            "## Missing boundary diagnostics",
            f"Tickets with missing/invalid boundaries: {len(exception_keys)}.",
        ]
    )
    if exception_sample:
        lines.append(f"Sample keys: {', '.join(exception_sample)}.")
    if not has_transitions:
        lines.extend(
            [
                "",
                "### PROMPT FOR THE USER",
                "Jira transition data was not available in the raw export. "
                "Please provide Jira transition timestamps or confirm that the fallback "
                "boundary inference rules should remain authoritative.",
            ]
        )
    DISTRIBUTION_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Canvas:
    def __init__(self, width: int, height: int, background: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 3)
        self.fill(background)

    def fill(self, color: tuple[int, int, int]) -> None:
        r, g, b = color
        for i in range(0, len(self.pixels), 3):
            self.pixels[i] = r
            self.pixels[i + 1] = g
            self.pixels[i + 2] = b

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.pixels[idx] = color[0]
            self.pixels[idx + 1] = color[1]
            self.pixels[idx + 2] = color[2]

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(y0, y1):
            for x in range(x0, x1):
                self.set_pixel(x, y, color)


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10001", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
}


def draw_text(canvas: Canvas, text: str, x: int, y: int, scale: int = 2, color: tuple[int, int, int] = (0, 0, 0)) -> None:
    cursor_x = x
    for char in text.upper():
        glyph = FONT.get(char, FONT[" "])
        for row_idx, row in enumerate(glyph):
            for col_idx, pixel in enumerate(row):
                if pixel == "1":
                    for sy in range(scale):
                        for sx in range(scale):
                            canvas.set_pixel(
                                cursor_x + col_idx * scale + sx,
                                y + row_idx * scale + sy,
                                color,
                            )
        cursor_x += (len(glyph[0]) + 1) * scale


def write_png(path: Path, canvas: Canvas) -> None:
    raw = bytearray()
    width = canvas.width
    height = canvas.height
    for y in range(height):
        raw.append(0)
        row_start = y * width * 3
        raw.extend(canvas.pixels[row_start : row_start + width * 3])
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", compressed))
        handle.write(chunk(b"IEND", b""))


def plot_histogram(samples: list[float], path: Path, title: str) -> None:
    width, height = 800, 600
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 60, 80
    canvas = Canvas(width, height)

    if not samples:
        draw_text(canvas, title, 40, 20, scale=2)
        draw_text(canvas, "NO DATA", 300, 280, scale=3, color=(200, 0, 0))
        write_png(path, canvas)
        return

    min_val = min(samples)
    max_val = max(samples)
    if min_val == max_val:
        max_val = min_val + 1
    bins = max(10, int(math.sqrt(len(samples))))
    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    for value in samples:
        idx = int((value - min_val) / bin_width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Axes
    canvas.draw_line(margin_left, margin_top, margin_left, margin_top + plot_height, (0, 0, 0))
    canvas.draw_line(
        margin_left,
        margin_top + plot_height,
        margin_left + plot_width,
        margin_top + plot_height,
        (0, 0, 0),
    )

    bar_width = plot_width / bins
    for idx, count in enumerate(counts):
        bar_height = int((count / max_count) * (plot_height - 10))
        x0 = int(margin_left + idx * bar_width)
        x1 = int(margin_left + (idx + 1) * bar_width - 2)
        y1 = margin_top + plot_height
        y0 = y1 - bar_height
        canvas.fill_rect(x0, y0, x1, y1, (66, 135, 245))

    draw_text(canvas, title, 40, 20, scale=2)
    draw_text(canvas, "DURATION (HOURS)", margin_left + 120, height - 50, scale=2)
    draw_text(canvas, "COUNT", 10, margin_top + 10, scale=2)

    write_png(path, canvas)


def main() -> None:
    jira_rows, jira_header, jira_path = load_jira_rows()
    pr_rows = load_pr_rows()

    key_field = "key" if "key" in jira_header else "issueKey"
    created_field = "fields.created" if "fields.created" in jira_header else "created"
    resolution_field = (
        "fields.resolutiondate" if "fields.resolutiondate" in jira_header else "resolutiondate"
    )

    pr_index = build_pr_index(pr_rows)
    phases = compute_phase_durations(jira_rows, pr_index, key_field, created_field, resolution_field)

    write_phase_durations(phases)

    phase_samples = {
        "dev": [p.dev_duration_hours for p in phases if p.dev_duration_hours is not None],
        "review": [p.review_duration_hours for p in phases if p.review_duration_hours is not None],
        "testing": [p.testing_duration_hours for p in phases if p.testing_duration_hours is not None],
    }

    best_fits = {}
    for phase, samples in phase_samples.items():
        best_fit, _ = select_best_fit(samples)
        best_fits[phase] = best_fit

    write_distribution_summary(phase_samples, best_fits)

    exception_keys = [p.key for p in phases if p.exception_reason]
    exception_sample = exception_keys[:10]
    has_transitions = any("transition" in (col or "").lower() for col in jira_header)

    write_distribution_summary_md(
        phase_samples,
        best_fits,
        exception_keys,
        exception_sample,
        jira_path,
        has_transitions,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_histogram(phase_samples["dev"], PLOTS_DIR / "dev_phase_histogram.png", "DEV PHASE HISTOGRAM")
    plot_histogram(
        phase_samples["review"],
        PLOTS_DIR / "review_phase_histogram.png",
        "REVIEW PHASE HISTOGRAM",
    )
    plot_histogram(
        phase_samples["testing"],
        PLOTS_DIR / "testing_phase_histogram.png",
        "TESTING PHASE HISTOGRAM",
    )


if __name__ == "__main__":
    main()
