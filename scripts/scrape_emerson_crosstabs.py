"""
Scrape demographic composition data from Emerson College poll articles.

For each Emerson article:
  1. Fetch the article HTML and extract the Google Sheets URL.
  2. Download the Sheet as CSV (public export).
  3. Parse the demographics section to extract xt_* composition values.
  4. Match to polls in polls_2026.csv by state + n_sample.
  5. Update polls_2026.csv with xt_* columns (idempotent).

Demographic → xt_* mapping:
  xt_race_white        = "White or Caucasian" Valid Percent / 100
  xt_race_black        = "Black or African American" / 100
  xt_race_hispanic     = "Hispanic or Latino of any race" / 100
  xt_race_asian        = "Asian" / 100
  xt_age_senior        = (60-69 + 70 or more) / 100
  xt_education_college = (College graduate + Postgraduate or higher) / 100
  xt_education_noncollege = 1 - xt_education_college
  xt_urbanicity_urban, xt_urbanicity_rural, xt_religion_evangelical: not available

Usage:
    uv run python scripts/scrape_emerson_crosstabs.py
    uv run python scripts/scrape_emerson_crosstabs.py --dry-run
    uv run python scripts/scrape_emerson_crosstabs.py --url URL [--url URL ...]
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"

USER_AGENT = "WetherVane-PollScraper/1.0 (political research)"
REQUEST_DELAY = 1.5  # seconds between HTTP requests

# xt_* columns the system supports (order matters for CSV column consistency).
# Urbanicity and religion are NOT available from Emerson — left blank.
XT_COLUMNS = [
    "xt_education_college",
    "xt_education_noncollege",
    "xt_race_white",
    "xt_race_black",
    "xt_race_hispanic",
    "xt_race_asian",
    "xt_urbanicity_urban",
    "xt_urbanicity_rural",
    "xt_age_senior",
    "xt_religion_evangelical",
]

# Known Emerson article URLs for 2026 races.
# Each article publishes a Google Sheets crosstab with a demographics section.
KNOWN_ARTICLE_URLS: list[str] = [
    "https://emersoncollegepolling.com/florida-2026-poll-donalds-leads-gop-primary-for-governor-republicans-outpace-democrats-in-florida-elections/",
    "https://emersoncollegepolling.com/georgia-2026-poll-senator-ossoff-starts-re-election-near-50-and-outpaces-gop-field/",
    "https://emersoncollegepolling.com/maine-2026-poll-platner-leads-gov-mills-democrats-lead-sen-collins-in-maine/",
    "https://emersoncollegepolling.com/texas-2026-primary-poll-talarico-paxton-with-narrow-edges-in-senate-primaries/",
]

# Map from Emerson article URL → state abbreviation.
# The state is used to match polls in the CSV.
# Note: one article may cover multiple races (same state, same survey).
URL_STATE_MAP: dict[str, str] = {
    "florida": "FL",
    "georgia": "GA",
    "maine": "ME",
    "texas": "TX",
    "arizona": "AZ",
    "michigan": "MI",
    "minnesota": "MN",
    "nevada": "NV",
    "north-carolina": "NC",
    "new-hampshire": "NH",
    "ohio": "OH",
    "pennsylvania": "PA",
    "wisconsin": "WI",
    "iowa": "IA",
    "oregon": "OR",
    "massachusetts": "MA",
    "alabama": "AL",
    "new-york": "NY",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 30) -> requests.Response:
    """GET a URL with standard headers. Raises on HTTP error."""
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Article → Google Sheets URL
# ---------------------------------------------------------------------------

# Regex that matches the spreadsheet ID inside a Google Sheets URL.
# Handles both /edit and /export patterns.
_SHEETS_PATTERN = re.compile(
    r"docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)"
)


def fetch_article(url: str) -> str:
    """Fetch the HTML of an Emerson article page. Returns the raw HTML."""
    logger.info("Fetching article: %s", url)
    resp = _get(url)
    return resp.text


def extract_sheet_url(html: str) -> Optional[str]:
    """
    Search the article HTML for a Google Sheets link and return the export CSV URL.

    Returns None if no Sheets link is found.
    """
    match = _SHEETS_PATTERN.search(html)
    if not match:
        return None
    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def infer_state_from_url(article_url: str) -> Optional[str]:
    """
    Infer the two-letter state abbreviation from the article URL slug.

    The Emerson URL contains a state name in the slug, e.g.:
      /florida-2026-poll-... → FL
    """
    slug = article_url.lower()
    for state_slug, abbr in URL_STATE_MAP.items():
        # Match whole-word (bounded by dashes or slash) to avoid false positives
        # like "new-york" matching "york" or "north" matching "north-carolina".
        if re.search(r"[/-]" + re.escape(state_slug) + r"[-/]", slug):
            return abbr
    return None


# ---------------------------------------------------------------------------
# Google Sheet CSV download
# ---------------------------------------------------------------------------

def download_sheet_csv(export_url: str) -> str:
    """Download a Google Sheets export CSV and return the raw text."""
    logger.info("Downloading sheet: %s", export_url)
    resp = _get(export_url)
    return resp.text


# ---------------------------------------------------------------------------
# Demographics parsing
# ---------------------------------------------------------------------------

def _parse_csv_lines(csv_text: str) -> list[list[str]]:
    """Parse CSV text into a list of rows (list of cell strings)."""
    reader = csv.reader(io.StringIO(csv_text))
    return list(reader)


def _find_section_start(rows: list[list[str]], header_fragment: str) -> Optional[int]:
    """
    Find the row index where a section header (case-insensitive substring match)
    appears. Returns None if not found.
    """
    needle = header_fragment.lower()
    for i, row in enumerate(rows):
        # Section headers are in column 0 (or span the whole row).
        cell = row[0].lower() if row else ""
        if needle in cell:
            return i
    return None


def _extract_pct_value(
    rows: list[list[str]], start: int, label_fragment: str, exact: bool = False
) -> Optional[float]:
    """
    Starting from row `start`, scan the next 20 rows for a row whose column-1
    cell contains `label_fragment` (case-insensitive). Return the Valid Percent
    column (column 3) as a float, or None if not found.

    If `exact=True`, the column-1 cell must equal `label_fragment` exactly
    (after stripping and lowercasing) rather than just containing it.
    """
    needle = label_fragment.lower()
    for row in rows[start : start + 20]:
        cell = row[1].lower().strip() if len(row) > 1 else ""
        matched = (cell == needle) if exact else (needle in cell)
        if matched:
            try:
                return float(row[3].strip()) if len(row) > 3 else None
            except (ValueError, IndexError):
                return None
    return None


def _sum_pct_values(
    rows: list[list[str]], start: int, label_fragments: list[str]
) -> Optional[float]:
    """
    Sum Valid Percent values for multiple label fragments within a section.
    Returns None if none of the labels are found.
    """
    total: Optional[float] = None
    for fragment in label_fragments:
        val = _extract_pct_value(rows, start, fragment)
        if val is not None:
            total = (total or 0.0) + val
    return total


def parse_demographics(csv_text: str) -> dict[str, Optional[float]]:
    """
    Parse an Emerson Google Sheets export CSV and extract xt_* demographic values.

    Returns a dict mapping xt_* field names to float values in [0, 1], or None
    if the corresponding data was not found in the sheet.

    All Emerson sheets have:
      - An ethnicity/race section (keyword: "ethnicity" or "Race")
      - An age section (keyword: "age range" or "Age,")
      - An education section (keyword: "education")

    The age breakdown varies across state polls:
      - Some use 18-29/30-39/.../70+ (FL/TX)
      - Some use 18-39/40-49/.../70+   (GA/ME)
    We sum 60-69 and 70+ regardless of how the younger bands are grouped.

    ME uses a simplified race format ("White" / "Non-white") rather than the
    full ethnicity breakdown. We handle both formats.
    """
    rows = _parse_csv_lines(csv_text)

    result: dict[str, Optional[float]] = {col: None for col in XT_COLUMNS}

    # --- Race / Ethnicity ---
    # Format A: "For statistical purposes only, can you please tell me your ethnicity?"
    ethnicity_start = _find_section_start(rows, "ethnicity")
    # Format B: bare "Race," header
    race_start = _find_section_start(rows, "race,")
    if race_start is None:
        # Some sheets spell it "Race\r" or just "Race" with no comma
        race_start = _find_section_start(rows, "race")
        # Avoid matching "What is your age range?" — check it's a short header
        if race_start is not None:
            cell = rows[race_start][0].strip().lower()
            if len(cell) > 10 and "age" in cell:
                race_start = None

    if ethnicity_start is not None:
        # Full breakdown available — keep None for genuinely missing values.
        # Use exact=True for labels that are substrings of other labels
        # (e.g., "asian" is a substring of "caucasian").
        for key in ("xt_race_white", "xt_race_black", "xt_race_hispanic", "xt_race_asian"):
            label, exact = _XT_LABEL_MAP[key]
            val = _extract_pct_value(rows, ethnicity_start, label, exact=exact)
            result[key] = val / 100 if val is not None else None

    elif race_start is not None:
        # Simplified format: only White / Non-white available.
        # Use exact=True so "white" doesn't match "non-white" first.
        white_pct = _extract_pct_value(rows, race_start, "white", exact=True)
        result["xt_race_white"] = white_pct / 100 if white_pct is not None else None
        # Other xt_race_* remain None — we don't have the breakdown

    # --- Age ---
    age_start = _find_section_start(rows, "age range")
    if age_start is None:
        # Some sheets use bare "Age," header
        age_start = _find_section_start(rows, "age,")
        if age_start is None:
            # Fallback: header is just "Age" with no comma on the same row
            for i, row in enumerate(rows):
                cell = row[0].strip().lower()
                if cell == "age":
                    age_start = i
                    break

    if age_start is not None:
        senior_60_69 = _extract_pct_value(rows, age_start, "60-69")
        senior_70_plus = _extract_pct_value(rows, age_start, "70 or more")
        if senior_60_69 is not None or senior_70_plus is not None:
            result["xt_age_senior"] = (
                (senior_60_69 or 0.0) + (senior_70_plus or 0.0)
            ) / 100

    # --- Education ---
    edu_start = _find_section_start(rows, "education you have attained")
    if edu_start is None:
        edu_start = _find_section_start(rows, "level of education")
    if edu_start is None:
        edu_start = _find_section_start(rows, "education")
        # Avoid matching "Education" rows that are part of crosstab data
        if edu_start is not None:
            # The section header should be in column 0, not col 1
            if rows[edu_start][0].strip().lower().startswith(","):
                edu_start = None

    if edu_start is not None:
        college_grad = _extract_pct_value(rows, edu_start, "college graduate")
        postgrad = _extract_pct_value(rows, edu_start, "postgraduate")
        if college_grad is not None or postgrad is not None:
            college_total = ((college_grad or 0.0) + (postgrad or 0.0)) / 100
            result["xt_education_college"] = college_total
            result["xt_education_noncollege"] = round(1.0 - college_total, 6)

    return result


# Mapping from xt_* key to (label_fragment, exact_match) for _extract_pct_value.
# "asian" must be exact because it is a substring of "caucasian".
# "white or caucasian" is not exact because there's no ambiguity.
_XT_LABEL_MAP: dict[str, tuple[str, bool]] = {
    "xt_race_white": ("white or caucasian", False),
    "xt_race_black": ("black or african american", False),
    "xt_race_hispanic": ("hispanic or latino", False),
    "xt_race_asian": ("asian", True),
}


# ---------------------------------------------------------------------------
# Poll CSV matching and updating
# ---------------------------------------------------------------------------

def load_polls_csv() -> tuple[list[str], list[dict]]:
    """
    Load polls_2026.csv and return (fieldnames, rows).

    Adds xt_* columns to fieldnames if not already present.
    """
    with POLLS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Ensure all xt_* columns exist in the header
    for col in XT_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row[col] = ""

    return fieldnames, rows


def _n_sample_from_row(row: dict) -> Optional[float]:
    """Parse n_sample from a poll row. Returns None if missing/invalid."""
    try:
        return float(row.get("n_sample", "") or "")
    except ValueError:
        return None


def match_and_update_polls(
    rows: list[dict],
    state: str,
    demographics: dict[str, Optional[float]],
    n_sample: Optional[float] = None,
    dry_run: bool = False,
) -> int:
    """
    Update all rows matching (pollster="Emerson College", geography=state) with
    xt_* values from `demographics`.

    If `n_sample` is provided, only update rows whose n_sample matches within ±5.

    Returns the number of rows updated.
    """
    updated = 0
    for row in rows:
        if row.get("pollster") != "Emerson College":
            continue
        if row.get("geography") != state:
            continue

        # Optional n_sample matching to distinguish between different surveys
        # from the same pollster and state (rare but possible)
        if n_sample is not None:
            row_n = _n_sample_from_row(row)
            if row_n is not None and abs(row_n - n_sample) > 5:
                logger.debug(
                    "Skipping row (n_sample mismatch): expected ~%.0f got %.0f, %s",
                    n_sample, row_n, row.get("race"),
                )
                continue

        # Apply xt_* values — only overwrite if new value is non-None
        any_new = False
        for col, val in demographics.items():
            if val is not None:
                new_str = f"{val:.6f}"
                if row.get(col) != new_str:
                    any_new = True
                if not dry_run:
                    row[col] = new_str

        if any_new:
            logger.info(
                "%s  %s → %s  %s updated",
                "DRY-RUN" if dry_run else "UPDATE",
                state,
                row.get("race"),
                row.get("date"),
            )
            updated += 1
        else:
            logger.debug("Already up-to-date: %s %s", state, row.get("race"))

    return updated


def save_polls_csv(fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows back to polls_2026.csv preserving column order."""
    with POLLS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows to %s", len(rows), POLLS_CSV)


# ---------------------------------------------------------------------------
# Article discovery
# ---------------------------------------------------------------------------

# Base URL for the category that lists state polls
_CATEGORY_BASE = "https://emersoncollegepolling.com/category/state-polls/"
# Regex for Emerson article URLs pointing to 2026 state poll articles
_ARTICLE_URL_PATTERN = re.compile(
    r'https://emersoncollegepolling\.com/[a-z0-9-]+-2026-poll[^"\']*'
)


def discover_article_urls(max_pages: int = 3) -> list[str]:
    """
    Scrape the Emerson state-polls category page(s) to discover poll article URLs.

    Returns a deduplicated list of article URLs containing "2026-poll" in the slug.
    Supplements (does not replace) KNOWN_ARTICLE_URLS.
    """
    found: set[str] = set(KNOWN_ARTICLE_URLS)

    for page_num in range(1, max_pages + 1):
        url = _CATEGORY_BASE if page_num == 1 else f"{_CATEGORY_BASE}page/{page_num}/"
        try:
            resp = _get(url)
            matches = _ARTICLE_URL_PATTERN.findall(resp.text)
            for m in matches:
                found.add(m.rstrip("/") + "/")
            logger.info("Category page %d: found %d total URLs so far", page_num, len(found))
            time.sleep(REQUEST_DELAY)
        except requests.HTTPError as exc:
            logger.warning("Category page %d returned %s — stopping.", page_num, exc)
            break

    return sorted(found)


# ---------------------------------------------------------------------------
# Per-article pipeline
# ---------------------------------------------------------------------------

def _extract_n_sample_from_sheet(csv_text: str) -> Optional[float]:
    """
    Try to infer the survey n_sample from the sheet by finding a "Total" row
    in the demographics section. Returns None if not found.
    """
    rows = _parse_csv_lines(csv_text)
    for row in rows:
        # Look for rows like: ,Total,NNNN,100.0
        if len(row) >= 4 and row[1].strip().lower() == "total":
            try:
                val = float(row[2].strip())
                if val > 100:  # Total sample size, not a percentage
                    return val
            except ValueError:
                continue
    return None


def process_article(
    article_url: str,
    rows: list[dict],
    dry_run: bool = False,
) -> int:
    """
    Full pipeline for a single Emerson article:
      1. Fetch article HTML.
      2. Extract Google Sheets export URL.
      3. Download sheet CSV.
      4. Parse demographics.
      5. Match and update poll rows.

    Returns the number of rows updated (0 on failure or no match).
    """
    state = infer_state_from_url(article_url)
    if state is None:
        logger.warning("Could not infer state from URL: %s", article_url)
        return 0

    # Fetch article HTML
    try:
        html = fetch_article(article_url)
        time.sleep(REQUEST_DELAY)
    except requests.RequestException as exc:
        logger.error("Failed to fetch article %s: %s", article_url, exc)
        return 0

    # Extract Sheets URL
    export_url = extract_sheet_url(html)
    if export_url is None:
        logger.warning("No Google Sheets link found in: %s", article_url)
        return 0
    logger.info("Found sheet for %s: %s", state, export_url)

    # Download sheet CSV
    try:
        csv_text = download_sheet_csv(export_url)
        time.sleep(REQUEST_DELAY)
    except requests.RequestException as exc:
        logger.error("Failed to download sheet for %s: %s", state, exc)
        return 0

    # Parse demographics
    demographics = parse_demographics(csv_text)
    populated = {k: v for k, v in demographics.items() if v is not None}
    if not populated:
        logger.warning("No demographic data parsed from sheet for %s", state)
        return 0
    logger.info("Parsed demographics for %s: %s", state, {
        k: f"{v:.3f}" for k, v in populated.items()
    })

    # Infer n_sample from sheet to help with matching
    n_sample = _extract_n_sample_from_sheet(csv_text)
    if n_sample:
        logger.info("Sheet n_sample: %.0f", n_sample)

    # Update poll rows
    updated = match_and_update_polls(rows, state, demographics, n_sample=n_sample, dry_run=dry_run)
    logger.info("Matched %d poll rows for %s", updated, state)
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Emerson College poll crosstabs and update polls_2026.csv"
    )
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        metavar="URL",
        help="Emerson article URL to process (can be repeated; default: all known URLs)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover additional article URLs from emersoncollegepolling.com/category/state-polls/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing to CSV",
    )
    args = parser.parse_args(argv)

    article_urls = list(args.urls or KNOWN_ARTICLE_URLS)

    if args.discover:
        article_urls = discover_article_urls()
        logger.info("Discovered %d article URLs", len(article_urls))

    # Load polls CSV once; update rows in memory; save once at the end
    fieldnames, rows = load_polls_csv()
    total_updated = 0

    for url in article_urls:
        updated = process_article(url, rows, dry_run=args.dry_run)
        total_updated += updated

    if not args.dry_run and total_updated > 0:
        save_polls_csv(fieldnames, rows)
        logger.info("Done. Updated %d poll rows total.", total_updated)
    elif args.dry_run:
        logger.info("Dry-run complete. Would update %d poll rows.", total_updated)
    else:
        logger.info("No rows updated.")

    return 0 if total_updated >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
