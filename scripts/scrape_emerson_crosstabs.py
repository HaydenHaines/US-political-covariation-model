"""
Scrape demographic composition data from Emerson College poll articles.

For each Emerson article:
  1. Fetch the article HTML and extract the Google Sheets URL.
  2. Download the Sheet as CSV (public export).
  3. Parse the demographics section to extract xt_* composition values.
  4. Discover the crosstab tab (second sheet) and parse per-group vote shares.
  5. Match to polls in polls_2026.csv by state + n_sample.
  6. Update polls_2026.csv with xt_* columns (idempotent).

Demographic → xt_* mapping:
  xt_race_white        = "White or Caucasian" Valid Percent / 100
  xt_race_black        = "Black or African American" / 100
  xt_race_hispanic     = "Hispanic or Latino of any race" / 100
  xt_race_asian        = "Asian" / 100
  xt_age_senior        = (60-69 + 70 or more) / 100
  xt_education_college = (College graduate + Postgraduate or higher) / 100
  xt_education_noncollege = 1 - xt_education_college
  xt_urbanicity_urban, xt_urbanicity_rural, xt_religion_evangelical: not available

Crosstab vote share → xt_vote_* mapping (from second sheet tab):
  xt_vote_race_white        = Dem vote share among white respondents
  xt_vote_race_black        = Dem vote share among Black respondents
  xt_vote_race_hispanic     = Dem vote share among Hispanic respondents
  xt_vote_race_asian        = Dem vote share among Asian respondents
  xt_vote_education_college = Dem vote share among college-educated respondents
  xt_vote_education_noncollege = Dem vote share among non-college respondents
  xt_vote_age_senior        = Dem vote share among 60+ respondents

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
    # Per-group vote share columns (from crosstab second tab).
    # These differ from the xt_* composition columns above: they encode what
    # fraction of each demographic group voted Dem, not how large the group is.
    "xt_vote_race_white",
    "xt_vote_race_black",
    "xt_vote_race_hispanic",
    "xt_vote_race_asian",
    "xt_vote_education_college",
    "xt_vote_education_noncollege",
    "xt_vote_age_senior",
]

# Substring patterns in crosstab row labels that identify the crosstab tab.
# The crosstab tab's row 2 (0-indexed) contains "Row N %" in answer-count columns.
_CROSSTAB_TAB_MARKER = "row n %"

# Maximum number of GIDs to try when searching for the crosstab tab.
# Sheets rarely have more than 5 tabs; 10 is a safe upper bound.
_MAX_GIDS_TO_TRY = 10

# Known Emerson article URLs for 2026 races.
# Each article publishes a Google Sheets crosstab with a demographics section.
KNOWN_ARTICLE_URLS: list[str] = [
    # FL (2026-04-02): Governor + Senate general election
    "https://emersoncollegepolling.com/florida-2026-poll-donalds-leads-gop-primary-for-governor-republicans-outpace-democrats-in-florida-elections/",
    # GA (2026-03-05): Senate general election
    "https://emersoncollegepolling.com/georgia-2026-poll-senator-ossoff-starts-re-election-near-50-and-outpaces-gop-field/",
    # ME (2026-03-26): Senate + Governor
    "https://emersoncollegepolling.com/maine-2026-poll-platner-leads-gov-mills-democrats-lead-sen-collins-in-maine/",
    # TX (2026-03-01): Primary polls (no general election match expected)
    "https://emersoncollegepolling.com/texas-2026-primary-poll-talarico-paxton-with-narrow-edges-in-senate-primaries/",
    # TX (2026-01-15): Governor + Senate general election
    "https://emersoncollegepolling.com/texas-2026-poll/",
    # AZ (2025-11-14): Governor
    "https://emersoncollegepolling.com/arizona-2026-governor/",
    # NV (2025-11-21): Governor
    "https://emersoncollegepolling.com/nevada-2026-poll/",
    # OH (2025-12-11): Governor + Senate
    "https://emersoncollegepolling.com/ohio-2026-poll-democrats-make-gains-in-races-for-governor-and-us-senate/",
    # MI (2026-01-29): Senate
    "https://emersoncollegepolling.com/michigan-2026-poll-crowded-democratic-senate-primary-remains-wide-open/",
    # MN (2026-02-11): Senate + Governor
    "https://emersoncollegepolling.com/minnesota-2026-poll-democrats-lead-gop-as-voters-cite-threats-to-democracy/",
    # NC (2025-08-01): Senate
    "https://emersoncollegepolling.com/north-carolina-2026-poll-cooper-starts-us-senate-race-with-six-point-lead-and-clear-name-recognition-advantage-over-whatley/",
    # NH (2026-03-23): Senate
    "https://emersoncollegepolling.com/new-hampshire-2026-sununu-leads-gop-nomination-ties-pappas-for-senate/",
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
# Crosstab tab discovery
# ---------------------------------------------------------------------------

def discover_sheet_gids(sheet_id: str) -> list[str]:
    """Discover all tab GIDs for a Google Sheet via the htmlview page.

    Google Sheets embeds tab GIDs as URL fragments in the htmlview page.
    We fetch that page and extract every unique GID with a regex.

    Args:
        sheet_id: The Google Sheets document ID (the long alphanumeric key
                  in the /spreadsheets/d/{ID}/ part of the URL).

    Returns:
        Deduplicated list of GID strings (e.g. ["0", "1234567890"]).
        Returns an empty list if the page cannot be fetched.
    """
    htmlview_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/htmlview"
    try:
        resp = _get(htmlview_url)
    except requests.RequestException as exc:
        logger.warning("Could not fetch htmlview for sheet %s: %s", sheet_id, exc)
        return []

    # GIDs appear as "gid=<digits>" in the HTML (both in anchor hrefs and
    # in the internal tab navigation markup).
    raw_gids = re.findall(r"gid=(\d+)", resp.text)

    # Deduplicate while preserving the order of first appearance.
    seen: set[str] = set()
    unique_gids: list[str] = []
    for gid in raw_gids:
        if gid not in seen:
            seen.add(gid)
            unique_gids.append(gid)

    logger.debug("Discovered GIDs for sheet %s: %s", sheet_id, unique_gids)
    return unique_gids


def download_sheet_tab_csv(sheet_id: str, gid: str) -> str:
    """Download a specific tab of a Google Sheet as CSV.

    Args:
        sheet_id: The Google Sheets document ID.
        gid: The tab GID string (digits only).

    Returns:
        Raw CSV text of that tab.
    """
    tab_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/export?format=csv&gid={gid}"
    )
    logger.debug("Downloading tab GID=%s for sheet %s", gid, sheet_id)
    resp = _get(tab_url)
    return resp.text


def identify_crosstab_gid(sheet_id: str, gids: list[str]) -> Optional[str]:
    """Try each GID and return the first one whose CSV looks like a crosstab tab.

    The crosstab tab is identified by the presence of "Row N %" in row 2
    (0-indexed) of its CSV.  The demographics/composition tab does NOT have
    this pattern — it uses "Valid Percent" style headers instead.

    Args:
        sheet_id: The Google Sheets document ID.
        gids: List of GID strings to try, in discovery order.

    Returns:
        The GID string of the crosstab tab, or None if none matches.
    """
    # Only check up to _MAX_GIDS_TO_TRY tabs to avoid hammering the server.
    for gid in gids[:_MAX_GIDS_TO_TRY]:
        try:
            csv_text = download_sheet_tab_csv(sheet_id, gid)
            time.sleep(REQUEST_DELAY)
        except requests.RequestException as exc:
            logger.debug("Tab GID=%s not accessible: %s", gid, exc)
            continue

        rows = _parse_csv_lines(csv_text)
        # Row 2 (index 2) is the "Count, Row N %" alternating header row.
        # A crosstab tab has at least 3 rows.
        if len(rows) < 3:
            continue

        # Check row 2 for the marker pattern — join all cells and search.
        row2_text = " ".join(rows[2]).lower()
        if _CROSSTAB_TAB_MARKER in row2_text:
            logger.info("Found crosstab tab: GID=%s for sheet %s", gid, sheet_id)
            return gid

    logger.debug("No crosstab tab found among GIDs %s", gids[:_MAX_GIDS_TO_TRY])
    return None


# ---------------------------------------------------------------------------
# Crosstab vote share parsing
# ---------------------------------------------------------------------------

# Map Emerson crosstab row labels to xt_vote_* column names.
# Each entry is (substring_to_match_in_lowercase_label, output_column_name).
# Order matters — more specific patterns must come before generic ones
# (e.g. "non-college" before "college") to prevent wrong matches.
_CROSSTAB_ROW_LABEL_MAP: list[tuple[str, str]] = [
    # Race / ethnicity
    ("white or caucasian", "xt_vote_race_white"),
    ("black or african american", "xt_vote_race_black"),
    ("hispanic or latino", "xt_vote_race_hispanic"),
    ("asian", "xt_vote_race_asian"),
    # Education — "non-college" variants must come before "college graduate"
    # to avoid accidentally matching "non-college" rows with the college key.
    ("non-college", "xt_vote_education_noncollege"),
    ("no college", "xt_vote_education_noncollege"),
    ("some college", "xt_vote_education_noncollege"),
    ("high school", "xt_vote_education_noncollege"),
    ("college graduate", "xt_vote_education_college"),
    ("postgraduate", "xt_vote_education_college"),
    # Age — 60+ buckets (we combine 60-69 and 70+ into the "senior" group)
    ("60-69", "xt_vote_age_senior"),
    ("70 or more", "xt_vote_age_senior"),
    ("70+", "xt_vote_age_senior"),
    ("65 or older", "xt_vote_age_senior"),
]


def _find_general_election_question(rows: list[list[str]]) -> Optional[str]:
    """Find the question text in row 0 that asks about the general election.

    Emerson crosstab CSVs have the question header text in row 0.  We look for
    the first non-empty cell that contains "general election" — this is the
    start of the column block we need to parse for vote shares.

    Returns the raw question text string, or None if not found.
    """
    if not rows:
        return None
    row0 = rows[0]
    for cell in row0:
        cell_stripped = cell.strip()
        if "general election" in cell_stripped.lower() and cell_stripped:
            return cell_stripped
    return None


def _infer_dem_candidate_column(
    rows: list[list[str]],
    question_text: str,
) -> Optional[int]:
    """Find the CSV column index containing the Dem candidate's "Row N %" value.

    Emerson crosstab CSVs have this layout:
      Row 0: Question headers (one header spans multiple columns).
      Row 1: Candidate names (one name per answer column).
      Row 2: Alternating "Count" / "Row N %" labels for each candidate.

    We need to find the "Row N %" column for the Democratic candidate so that
    we can extract the per-demographic vote share percentage.

    Strategy:
      1. Find the column in row 0 that contains the general election question.
      2. Read candidate names from row 1 starting at that column.
      3. Parse the question text for "(Democrat)" or similar party markers
         to identify which candidate is the Democrat.
      4. Return the index of their "Row N %" column in row 2.

    Args:
        rows: Parsed CSV rows (list of lists).
        question_text: The general election question text from row 0.

    Returns:
        Column index of the "Row N %" cell for the Democratic candidate,
        or None if we cannot determine it confidently.
    """
    if len(rows) < 3:
        return None

    row0 = rows[0]  # question headers
    row1 = rows[1]  # candidate names
    row2 = rows[2]  # "Count" / "Row N %" alternating labels

    # Find the column where the general election question header starts.
    question_col: Optional[int] = None
    question_lower = question_text.lower()
    for col_idx, cell in enumerate(row0):
        if cell.strip().lower() == question_lower:
            question_col = col_idx
            break

    if question_col is None:
        # Partial match fallback — the full question may not fit in one cell.
        for col_idx, cell in enumerate(row0):
            if "general election" in cell.lower() and cell.strip():
                question_col = col_idx
                break

    if question_col is None:
        return None

    # Collect candidate names for columns within this question's span.
    # The span continues until the next non-empty cell in row 0.
    candidate_cols: list[tuple[str, int]] = []
    for col_idx in range(question_col, len(row1)):
        # A new question starts when row0 has a fresh non-empty header.
        if col_idx > question_col and col_idx < len(row0) and row0[col_idx].strip():
            break
        name = row1[col_idx].strip() if col_idx < len(row1) else ""
        if name:
            candidate_cols.append((name, col_idx))

    if not candidate_cols:
        return None

    # Try to identify the Democratic candidate from the question text or candidate
    # name.  Emerson questions often read: "Democrat Jon Ossoff" or show "(D)" in
    # the candidate name cell.
    dem_candidate_col: Optional[int] = None
    qt_lower = question_text.lower()

    for name, col_idx in candidate_cols:
        name_lower = name.lower()

        # Direct "(D)" marker in the candidate name cell itself.
        if "(d)" in name_lower:
            dem_candidate_col = col_idx
            break

        # "Democrat" keyword near candidate's first name in the question text.
        if "democrat" in qt_lower:
            first_word = name_lower.split()[0] if name_lower.split() else ""
            name_pos = qt_lower.find(first_word) if first_word else -1
            dem_pos = qt_lower.find("democrat")
            # Within 60 characters is a strong signal they are co-referential.
            if name_pos >= 0 and abs(name_pos - dem_pos) < 60:
                dem_candidate_col = col_idx
                break

        # Explicit "(d)" marker in the question text immediately after the name.
        if f"{name.split()[0].lower()} (d)" in qt_lower:
            dem_candidate_col = col_idx
            break

    if dem_candidate_col is None:
        return None

    # Find the "Row N %" column (row 2) associated with this candidate column.
    # It is typically dem_candidate_col + 1 but we scan a small window to be safe.
    for offset in range(0, 3):
        check_col = dem_candidate_col + offset
        if check_col < len(row2) and "row n %" in row2[check_col].lower():
            return check_col

    return None


def parse_crosstab_vote_shares(csv_text: str) -> dict[str, float]:
    """Parse the crosstab tab CSV and extract per-group Democratic vote shares.

    The crosstab tab has this structure:
      Row 0: Question headers (one header spans multiple columns).
              We target the column block for the "general election" question.
      Row 1: Candidate names.
      Row 2: "Count" / "Row N %" alternating column labels.
      Rows 3+: Demographic group rows.
                Column 0 or 1: group label (e.g. "White or Caucasian").
                Other columns: Count / Row N % pairs for each candidate.

    We extract the "Row N %" value for the Democratic candidate for each
    demographic group and convert to a 0–1 scale.

    For demographic groups that appear in multiple rows mapping to the same
    xt_vote_* column (e.g. "College graduate" and "Postgraduate" both map to
    xt_vote_education_college), we average the two values since we don't have
    within-group sample sizes to do a proper weighted average.

    Args:
        csv_text: Raw CSV text of the crosstab tab.

    Returns:
        Dict mapping xt_vote_* column names to float values in [0, 1].
        Returns empty dict if parsing fails at any required step.
    """
    rows = _parse_csv_lines(csv_text)

    if len(rows) < 3:
        logger.debug("Crosstab CSV has fewer than 3 rows — cannot parse vote shares")
        return {}

    # Find the general election question to know which columns to read.
    question_text = _find_general_election_question(rows)
    if question_text is None:
        logger.debug("No 'general election' question found in crosstab row 0")
        return {}

    # Find the column index for the Democratic candidate's "Row N %" in row 2.
    dem_pct_col = _infer_dem_candidate_column(rows, question_text)
    if dem_pct_col is None:
        logger.debug("Could not identify Democratic candidate's 'Row N %%' column")
        return {}

    logger.debug(
        "Crosstab: question='%.60s...', dem_pct_col=%d", question_text, dem_pct_col
    )

    # Accumulate raw values before averaging.  Some xt_vote_* columns get
    # contributions from multiple row labels (e.g. senior = 60-69 + 70+).
    # We track (sum, count) and average at the end.
    accumulators: dict[str, list[float]] = {}

    for row in rows[3:]:
        # Group labels appear in column 1 in most layouts; column 0 in compact ones.
        label = ""
        if len(row) > 1 and row[1].strip():
            label = row[1].strip()
        elif row and row[0].strip():
            label = row[0].strip()

        if not label:
            continue  # blank separator row

        label_lower = label.lower()

        # Match against known demographic labels in priority order.
        for pattern, xt_col in _CROSSTAB_ROW_LABEL_MAP:
            if pattern not in label_lower:
                continue

            # Extract the "Row N %" percentage from the Democratic candidate column.
            if dem_pct_col >= len(row):
                break  # row is shorter than expected — skip

            cell = row[dem_pct_col].strip().rstrip("%")
            if not cell:
                break

            try:
                pct = float(cell)
            except ValueError:
                break

            # Convert percentage to 0–1 fraction and accumulate.
            vote_share = pct / 100.0
            accumulators.setdefault(xt_col, []).append(vote_share)
            break  # matched — move to next row

    # Average accumulated values for each column.
    result: dict[str, float] = {}
    for xt_col, values in accumulators.items():
        result[xt_col] = sum(values) / len(values)

    logger.debug("Parsed crosstab vote shares: %s", result)
    return result


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
    Starting from row `start`, scan the next 20 rows for a row whose label cell
    contains `label_fragment` (case-insensitive). Return the percent value as a
    float, or None if not found.

    If `exact=True`, the label cell must equal `label_fragment` exactly
    (after stripping and lowercasing) rather than just containing it.

    Handles two CSV column layouts:
      - Standard (FL/GA/ME/TX): blank col 0, label col 1, count col 2, pct col 3
      - Compact (OH): label col 0, count col 1, pct col 2 (with % suffix)
    """
    needle = label_fragment.lower()
    for row in rows[start : start + 20]:
        # Try standard layout first (label in col 1, pct in col 3)
        label_col1 = row[1].lower().strip() if len(row) > 1 else ""
        matched_col1 = (label_col1 == needle) if exact else (needle in label_col1)
        if matched_col1 and len(row) > 3:
            try:
                return float(row[3].strip().rstrip("%"))
            except (ValueError, IndexError):
                pass

        # Try compact layout (label in col 0, pct in col 2)
        label_col0 = row[0].lower().strip() if row else ""
        matched_col0 = (label_col0 == needle) if exact else (needle in label_col0)
        if matched_col0 and len(row) > 2:
            try:
                return float(row[2].strip().rstrip("%"))
            except (ValueError, IndexError):
                pass

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


def _extract_sheet_id_from_export_url(export_url: str) -> Optional[str]:
    """Extract the spreadsheet ID from a Google Sheets export URL.

    Export URLs look like:
      https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv

    Returns the SHEET_ID string, or None if the pattern does not match.
    """
    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)/", export_url)
    return match.group(1) if match else None


def process_article(
    article_url: str,
    rows: list[dict],
    dry_run: bool = False,
) -> int:
    """
    Full pipeline for a single Emerson article:
      1. Fetch article HTML.
      2. Extract Google Sheets export URL.
      3. Download sheet CSV (first tab — demographics).
      4. Parse demographics composition.
      5. Discover sheet GIDs and find the crosstab tab.
      6. Parse per-group vote shares from the crosstab tab.
      7. Merge vote shares into demographics dict.
      8. Match and update poll rows.

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

    # Download the first tab (demographics composition)
    try:
        csv_text = download_sheet_csv(export_url)
        time.sleep(REQUEST_DELAY)
    except requests.RequestException as exc:
        logger.error("Failed to download sheet for %s: %s", state, exc)
        return 0

    # Parse demographics composition (xt_* columns)
    demographics = parse_demographics(csv_text)
    populated = {k: v for k, v in demographics.items() if v is not None}
    if not populated:
        logger.warning("No demographic data parsed from sheet for %s", state)
        return 0
    logger.info("Parsed demographics for %s: %s", state, {
        k: f"{v:.3f}" for k, v in populated.items()
    })

    # Attempt to parse per-group vote shares from the crosstab tab.
    # This is best-effort — if the sheet has no second tab or we can't parse
    # it, we fall back gracefully to the composition-only data.
    sheet_id = _extract_sheet_id_from_export_url(export_url)
    if sheet_id is not None:
        gids = discover_sheet_gids(sheet_id)
        time.sleep(REQUEST_DELAY)

        if gids:
            crosstab_gid = identify_crosstab_gid(sheet_id, gids)
            if crosstab_gid is not None:
                try:
                    crosstab_csv = download_sheet_tab_csv(sheet_id, crosstab_gid)
                    time.sleep(REQUEST_DELAY)
                    vote_shares = parse_crosstab_vote_shares(crosstab_csv)
                    if vote_shares:
                        logger.info(
                            "Parsed crosstab vote shares for %s: %s",
                            state,
                            {k: f"{v:.3f}" for k, v in vote_shares.items()},
                        )
                        # Merge vote shares into the demographics dict.
                        # Both dicts have the same value type (Optional[float]).
                        demographics.update(vote_shares)
                    else:
                        logger.debug("No vote shares parsed from crosstab tab for %s", state)
                except requests.RequestException as exc:
                    logger.warning(
                        "Could not download crosstab tab for %s: %s", state, exc
                    )
            else:
                logger.debug("No crosstab tab found for %s", state)
        else:
            logger.debug("No GIDs discovered for sheet %s", sheet_id)
    else:
        logger.debug("Could not extract sheet ID from URL: %s", export_url)

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
