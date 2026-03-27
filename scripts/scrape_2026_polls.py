"""
Scrape 2026 election polls from Wikipedia and 270toWin.

Dual-source scraper targeting FL, GA, AL governor and Senate races.
Outputs to data/polls/polls_2026.csv in the project's standard schema.

Usage:
    uv run python scripts/scrape_2026_polls.py
    uv run python scripts/scrape_2026_polls.py --dry-run
    uv run python scripts/scrape_2026_polls.py --races "FL Governor,GA Senate"
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"

USER_AGENT = "WetherVane-PollScraper/1.0 (political research)"
REQUEST_DELAY = 2  # seconds between HTTP requests

# ---------------------------------------------------------------------------
# Race configuration
# ---------------------------------------------------------------------------
RACE_CONFIG = {
    "2026 FL Governor": {
        "state": "FL",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Florida_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/florida",
        # Known general-election candidate last names.
        # Multiple possible D/R nominees listed; scraper matches any.
        "dem_candidates": [
            "jolly",
            "demings",
            "cava",
            "levine cava",
            "daniella levine cava",
            "gwen graham",
            "graham",
        ],
        "rep_candidates": [
            "donalds",
            "desantis",
            "collins",
            "renner",
            "fishback",
            "simpson",
            "casey desantis",
        ],
    },
    "2026 FL Senate": {
        "state": "FL",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_special_election_in_Florida",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/florida",
        "dem_candidates": [
            "grayson",
            "nixon",
            "moskowitz",
            "vindman",
        ],
        "rep_candidates": [
            "moody",
            "lang",
        ],
    },
    "2026 GA Governor": {
        "state": "GA",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Georgia_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/georgia",
        "dem_candidates": [
            "bottoms",
            "keisha lance bottoms",
            "duncan",
        ],
        "rep_candidates": [
            "jones",
            "burt jones",
            "mike collins",
            "collins",
            "jackson",
        ],
    },
    "2026 GA Senate": {
        "state": "GA",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Georgia",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/georgia",
        "dem_candidates": [
            "ossoff",
            "jon ossoff",
        ],
        "rep_candidates": [
            "carter",
            "buddy carter",
            "mike collins",
            "collins",
            "dooley",
            "rich dooley",
        ],
    },
    "2026 AL Governor": {
        "state": "AL",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Alabama_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/alabama",
        "dem_candidates": [
            "jones",
            "doug jones",
            "flowers",
            "yolanda flowers",
        ],
        "rep_candidates": [
            "tuberville",
            "tommy tuberville",
            "mcfeeters",
        ],
    },
    "2026 AL Senate": {
        "state": "AL",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Alabama",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/alabama",
        "dem_candidates": [
            "figures",
            "shomari figures",
        ],
        "rep_candidates": [
            "marshall",
            "steve marshall",
            "moore",
            "barry moore",
            "hudson",
            "jared hudson",
            "dobson",
            "caroleene dobson",
        ],
    },
    # ── National competitive races (added S213) ──────────────────
    "2026 IA Senate": {
        "state": "IA",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Iowa",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/iowa",
        "dem_candidates": ["franken", "michael franken"],
        "rep_candidates": ["ernst", "joni ernst"],
    },
    "2026 ME Senate": {
        "state": "ME",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Maine",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/maine",
        "dem_candidates": ["gideon", "sara gideon", "pingree", "chellie pingree"],
        "rep_candidates": ["collins", "susan collins"],
    },
    "2026 MI Governor": {
        "state": "MI",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Michigan_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/michigan",
        "dem_candidates": ["gilchrist", "garlin gilchrist"],
        "rep_candidates": ["james", "john james", "soldano", "garrett soldano"],
    },
    "2026 MI Senate": {
        "state": "MI",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Michigan",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/michigan",
        "dem_candidates": ["peters", "gary peters", "slotkin", "elissa slotkin"],
        "rep_candidates": ["james", "john james", "kelley", "tudor dixon"],
    },
    "2026 MN Senate": {
        "state": "MN",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Minnesota",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/minnesota",
        "dem_candidates": ["smith", "tina smith"],
        "rep_candidates": ["jensen", "scott jensen", "birk", "matt birk"],
    },
    "2026 NC Senate": {
        "state": "NC",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_North_Carolina",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/north-carolina",
        "dem_candidates": ["jackson", "jeff jackson"],
        "rep_candidates": ["tillis", "thom tillis"],
    },
    "2026 NH Senate": {
        "state": "NH",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_New_Hampshire",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/new-hampshire",
        "dem_candidates": ["shaheen", "jeanne shaheen"],
        "rep_candidates": ["morse", "chuck morse", "sununu", "chris sununu"],
    },
    "2026 OH Governor": {
        "state": "OH",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Ohio_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/ohio",
        "dem_candidates": ["whaley", "nan whaley"],
        "rep_candidates": ["husted", "jon husted"],
    },
    "2026 OR Senate": {
        "state": "OR",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Oregon",
        "ttw_url": "https://www.270towin.com/2026-senate-polls/oregon",
        "dem_candidates": ["merkley", "jeff merkley"],
        "rep_candidates": ["drazan", "christine drazan"],
    },
    "2026 PA Governor": {
        "state": "PA",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Pennsylvania_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/pennsylvania",
        "dem_candidates": ["shapiro", "josh shapiro"],
        "rep_candidates": ["mastriano", "mccormick", "dave mccormick"],
    },
    "2026 TX Governor": {
        "state": "TX",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Texas_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/texas",
        "dem_candidates": ["allred", "colin allred", "casar", "greg casar"],
        "rep_candidates": ["patrick", "dan patrick", "paxton", "ken paxton"],
    },
    "2026 WI Governor": {
        "state": "WI",
        "wiki_url": "https://en.wikipedia.org/wiki/2026_Wisconsin_gubernatorial_election",
        "ttw_url": "https://www.270towin.com/2026-governor-polls/wisconsin",
        "dem_candidates": ["evers", "tony evers"],
        "rep_candidates": ["kleefisch", "rebecca kleefisch", "michels", "tim michels"],
    },
}

# ---------------------------------------------------------------------------
# Pollster name normalization
# ---------------------------------------------------------------------------
POLLSTER_ALIASES: dict[str, str] = {
    "emerson": "Emerson College",
    "emerson college": "Emerson College",
    "emerson college polling": "Emerson College",
    "emerson college polling society": "Emerson College",
    "quinnipiac": "Quinnipiac University",
    "quinnipiac university": "Quinnipiac University",
    "quinnipiac u.": "Quinnipiac University",
    "mason-dixon": "Mason-Dixon",
    "mason-dixon polling": "Mason-Dixon",
    "mason dixon": "Mason-Dixon",
    "mason dixon polling": "Mason-Dixon",
    "fox news": "FOX News",
    "fox news poll": "FOX News",
    "fox": "FOX News",
    "cnn": "CNN",
    "cnn/ssrs": "CNN/SSRS",
    "ssrs": "CNN/SSRS",
    "morning consult": "Morning Consult",
    "morningconsult": "Morning Consult",
    "unf": "University of North Florida",
    "unf poll": "University of North Florida",
    "university of north florida": "University of North Florida",
    "univ. of north florida": "University of North Florida",
    "uga": "University of Georgia",
    "university of georgia": "University of Georgia",
    "st. pete polls": "St. Pete Polls",
    "st pete polls": "St. Pete Polls",
    "cygnal": "Cygnal",
    "cygnal (r)": "Cygnal",
    "tyson group": "Tyson Group",
    "tyson": "Tyson Group",
    "the tyson group (r)": "Tyson Group",
    "tyson group (r)": "Tyson Group",
    "quantus insights": "Quantus Insights",
    "quantus": "Quantus Insights",
    "quantus insights (r)": "Quantus Insights",
    "atlasintel": "AtlasIntel",
    "atlas intel": "AtlasIntel",
    "trafalgar": "Trafalgar Group",
    "trafalgar group": "Trafalgar Group",
    "the trafalgar group": "Trafalgar Group",
    "trafalgar group (r)": "Trafalgar Group",
    "suffolk university": "Suffolk University",
    "suffolk": "Suffolk University",
    "suffolk university/usa today": "Suffolk University",
    "marist": "Marist College",
    "marist college": "Marist College",
    "marist poll": "Marist College",
    "nbc news/marist": "Marist College",
    "monmouth university": "Monmouth University",
    "monmouth": "Monmouth University",
    "siena college": "Siena College",
    "siena": "Siena College",
    "nyt/siena": "Siena College",
    "new york times/siena college": "Siena College",
    "echelon insights": "Echelon Insights",
    "echelon": "Echelon Insights",
    "public policy polling": "Public Policy Polling",
    "ppp": "Public Policy Polling",
    "insider advantage": "InsiderAdvantage",
    "insideradvantage": "InsiderAdvantage",
    "insideradvantage (r)": "InsiderAdvantage",
    "data for progress": "Data for Progress",
    "dfp": "Data for Progress",
    "change research": "Change Research",
    "surveyusa": "SurveyUSA",
    "survey usa": "SurveyUSA",
    "mclaughlin & associates": "McLaughlin & Associates",
    "mclaughlin": "McLaughlin & Associates",
    "mclaughlin & associates (r)": "McLaughlin & Associates",
    "rasmussen reports": "Rasmussen Reports",
    "rasmussen reports (r)": "Rasmussen Reports",
    "rasmussen": "Rasmussen Reports",
    "wpa intelligence (r)": "WPA Intelligence",
    "wpa intelligence": "WPA Intelligence",
    "remington research group (r)": "Remington Research Group",
    "remington research group": "Remington Research Group",
    "remington": "Remington Research Group",
    "jmc analytics": "JMC Analytics",
    "jmc analytics & polling": "JMC Analytics",
    "fabrizio, lee & associates (r)": "Fabrizio Lee & Associates",
    "fabrizio lee & associates": "Fabrizio Lee & Associates",
    "victory insights (r)": "Victory Insights",
    "victory insights": "Victory Insights",
    "bendixen & amandi international (d)": "Bendixen & Amandi International",
    "bendixen & amandi international": "Bendixen & Amandi International",
    "frederick polls (d)": "Frederick Polls",
    "frederick polls": "Frederick Polls",
    "plymouth union public research (r)": "Plymouth Union Public Research",
    "plymouth union public research": "Plymouth Union Public Research",
    "the alabama poll": "The Alabama Poll",
    "tipp insights": "TIPP Insights",
    "tipp": "TIPP Insights",
    "atlanta journal-constitution": "Atlanta Journal-Constitution",
    "ajc": "Atlanta Journal-Constitution",
}


def normalize_pollster(name: str) -> str:
    """Normalize pollster name to canonical form."""
    if not name or not isinstance(name, str):
        return str(name) if name else ""
    stripped = name.strip()
    key = stripped.lower().strip()
    # Remove trailing footnote markers like [1], [a], etc.
    key = re.sub(r"\[.*?\]", "", key).strip()
    if key in POLLSTER_ALIASES:
        return POLLSTER_ALIASES[key]
    # Return original (cleaned) if no alias found
    return re.sub(r"\[.*?\]", "", stripped).strip()


# ---------------------------------------------------------------------------
# Two-party share conversion
# ---------------------------------------------------------------------------
def two_party_share(dem_pct: float, rep_pct: float) -> float | None:
    """Convert raw D% and R% to two-party Democratic share.

    Returns None if D+R total is below 30% (too many undecided for meaningful
    two-party conversion) or if the result is outside the (0.15, 0.85) range.
    """
    if dem_pct <= 0 or rep_pct <= 0:
        return None
    total = dem_pct + rep_pct
    if total < 30:
        logger.warning(
            "D+R total %.1f%% too low for two-party conversion (D=%.1f, R=%.1f)",
            total,
            dem_pct,
            rep_pct,
        )
        return None
    share = dem_pct / total
    if share < 0.15 or share > 0.85:
        logger.warning(
            "Two-party share %.3f outside sanity range (D=%.1f, R=%.1f)", share, dem_pct, rep_pct
        )
        return None
    return round(share, 4)


# ---------------------------------------------------------------------------
# HTTP fetch helper
# ---------------------------------------------------------------------------
def fetch_html(url: str) -> str | None:
    """Fetch URL content with User-Agent header. Returns HTML string or None."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
def parse_poll_date(date_str: str) -> str | None:
    """Parse a poll date string and return YYYY-MM-DD (end date if range).

    Handles formats like:
      - "March 4, 2026"
      - "Mar 1-4, 2026"
      - "February 28 - March 4, 2026"
      - "2/28 - 3/4/2026"
      - "3/09/2026"
      - "2026-03-04"
    """
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()
    # Remove footnotes
    s = re.sub(r"\[.*?\]", "", s).strip()

    # Already ISO format
    iso_match = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if iso_match:
        return iso_match.group(1)

    # Try pandas date parser on the last date in a range
    # Split on common range separators
    for sep in ["\u2013", "\u2014", "-", " to "]:
        if sep in s:
            parts = s.split(sep)
            end_part = parts[-1].strip()
            # If end part is just a day number, prepend month from start
            if re.match(r"^\d{1,2},?\s*\d{4}$", end_part):
                # e.g. "March 1-4, 2026" -> end_part = "4, 2026"
                start_part = parts[0].strip()
                month_match = re.match(r"([A-Za-z]+)", start_part)
                if month_match:
                    end_part = f"{month_match.group(1)} {end_part}"
            elif re.match(r"^\d{1,2}$", end_part):
                # e.g. "March 1-4" with year elsewhere
                start_part = parts[0].strip()
                month_match = re.match(r"([A-Za-z]+)\s+\d", start_part)
                year_match = re.search(r"(\d{4})", s)
                if month_match and year_match:
                    end_part = f"{month_match.group(1)} {end_part}, {year_match.group(1)}"
            try:
                dt = pd.to_datetime(end_part, format="mixed", dayfirst=False)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

    # No range separator — try parsing the whole string
    try:
        dt = pd.to_datetime(s, format="mixed", dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        logger.debug("Could not parse date: %s", date_str)
        return None


# ---------------------------------------------------------------------------
# Extract numeric percentage from a cell value
# ---------------------------------------------------------------------------
def extract_pct(val) -> float | None:
    """Extract a numeric percentage from a table cell value."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Remove % sign and footnotes
    s = re.sub(r"\[.*?\]", "", s)
    s = s.replace("%", "").strip()
    try:
        v = float(s)
        if 0 < v < 100:
            return v
    except (ValueError, TypeError):
        pass
    return None


def extract_sample_size(val) -> int | None:
    """Extract integer sample size from a cell value."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Remove footnotes, parentheticals like (LV), (RV), commas
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace(",", "").strip()
    # Extract first integer
    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        if n >= 50:  # sanity: sample sizes below 50 are suspicious
            return n
    return None


# ---------------------------------------------------------------------------
# Column classification using candidate names
# ---------------------------------------------------------------------------
def _classify_columns(
    df: pd.DataFrame,
    dem_candidates: list[str],
    rep_candidates: list[str],
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Classify columns into pollster, date, sample, dem_col, rep_col.

    Uses known candidate names per race to identify D/R percentage columns.
    Returns (pollster_col, date_col, sample_col, dem_col, rep_col).
    """
    cols = list(df.columns)
    cols_lower = [str(c).lower() for c in cols]

    # Find structural columns
    pollster_col = None
    date_col = None
    sample_col = None

    # Collect all source/poll columns — 270toWin splits "Source" into
    # "Source" (NaN) and "Source.1" (actual pollster name).  Pick the last
    # one that has non-null data.
    source_candidates = []
    for i, cl in enumerate(cols_lower):
        if any(kw in cl for kw in ["poll", "source"]):
            source_candidates.append(cols[i])
        elif date_col is None and "date" in cl:
            date_col = cols[i]
        elif sample_col is None and any(kw in cl for kw in ["sample", "size"]):
            sample_col = cols[i]

    # Pick the source column with the most non-null string values
    if source_candidates:
        best_col = source_candidates[0]
        best_count = 0
        for sc in source_candidates:
            non_null = df[sc].dropna().astype(str).apply(lambda x: x.strip() != "").sum()
            if non_null > best_count:
                best_count = non_null
                best_col = sc
        pollster_col = best_col

    # Find D and R columns by matching candidate names in column headers
    dem_col = None
    rep_col = None

    for c in cols:
        c_lower = str(c).lower()
        # Check explicit party labels first
        if any(tag in c_lower for tag in ["(d)", "democrat"]):
            dem_col = c
            continue
        if any(tag in c_lower for tag in ["(r)", "republican"]):
            rep_col = c
            continue
        # Check against known candidate names
        for name in dem_candidates:
            if name.lower() in c_lower:
                dem_col = c
                break
        for name in rep_candidates:
            if name.lower() in c_lower:
                rep_col = c
                break

    return pollster_col, date_col, sample_col, dem_col, rep_col


def _is_general_election_table(
    df: pd.DataFrame,
    dem_candidates: list[str],
    rep_candidates: list[str],
) -> bool:
    """Check if a table is a general election matchup (has both D and R candidate columns)."""
    cols_lower = [str(c).lower() for c in df.columns]
    has_dem = False
    has_rep = False
    for cl in cols_lower:
        if any(tag in cl for tag in ["(d)", "democrat"]):
            has_dem = True
        if any(tag in cl for tag in ["(r)", "republican"]):
            has_rep = True
        for name in dem_candidates:
            if name.lower() in cl:
                has_dem = True
        for name in rep_candidates:
            if name.lower() in cl:
                has_rep = True
    return has_dem and has_rep


def _has_pollster_column(df: pd.DataFrame) -> bool:
    """Check if a table has a pollster/source column."""
    cols_lower = [str(c).lower() for c in df.columns]
    return any("poll" in c or "source" in c for c in cols_lower)


# ---------------------------------------------------------------------------
# Generic table parser (shared by Wikipedia and 270toWin)
# ---------------------------------------------------------------------------
def _parse_poll_table(
    df: pd.DataFrame,
    race_label: str,
    source_name: str,
    dem_candidates: list[str],
    rep_candidates: list[str],
) -> list[dict]:
    """Parse a poll table DataFrame into poll dicts."""
    pollster_col, date_col, sample_col, dem_col, rep_col = _classify_columns(
        df, dem_candidates, rep_candidates
    )

    if not dem_col or not rep_col:
        return []

    polls = []
    for _, row in df.iterrows():
        # Extract pollster
        pollster_raw = ""
        if pollster_col:
            pollster_raw = str(row.get(pollster_col, ""))
        if not pollster_raw or pollster_raw == "nan":
            continue
        # Skip header/footer/aggregate rows
        if any(
            kw in pollster_raw.lower()
            for kw in [
                "polling",
                "average",
                "rcp",
                "aggregate",
                "final result",
                "270towin",
                "realclearpolitics",
                "race to the wh",
            ]
        ):
            continue

        date_raw = str(row.get(date_col, "")) if date_col else ""
        date_parsed = parse_poll_date(date_raw)

        sample_raw = row.get(sample_col) if sample_col else None
        n_sample = extract_sample_size(sample_raw)

        # Extract sample type (LV/RV)
        sample_type = ""
        if sample_raw and isinstance(sample_raw, str):
            s_upper = sample_raw.upper()
            if "LV" in s_upper:
                sample_type = "LV"
            elif "RV" in s_upper:
                sample_type = "RV"

        dem_pct = extract_pct(row.get(dem_col))
        rep_pct = extract_pct(row.get(rep_col))

        if dem_pct is None or rep_pct is None:
            continue

        dem_share = two_party_share(dem_pct, rep_pct)
        if dem_share is None:
            continue

        polls.append(
            {
                "race": race_label,
                "pollster_raw": pollster_raw.strip(),
                "pollster": normalize_pollster(pollster_raw),
                "date": date_parsed,
                "n_sample": n_sample,
                "dem_pct": dem_pct,
                "rep_pct": rep_pct,
                "dem_share": dem_share,
                "source": source_name,
                "sample_type": sample_type,
            }
        )

    return polls


# ---------------------------------------------------------------------------
# Wikipedia scraper
# ---------------------------------------------------------------------------
def scrape_wikipedia(
    race_label: str,
    url: str,
    dem_candidates: list[str],
    rep_candidates: list[str],
) -> list[dict]:
    """Scrape poll data from a Wikipedia election article."""
    logger.info("  Wikipedia: %s", url)
    html = fetch_html(url)
    if not html:
        return []

    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception as e:
        logger.error("  Failed to parse tables from %s: %s", url, e)
        return []

    logger.info("  Found %d tables total", len(tables))

    all_polls = []
    for idx, df in enumerate(tables):
        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(str(x) for x in col if "Unnamed" not in str(x)).strip()
                for col in df.columns
            ]

        if not _has_pollster_column(df):
            continue

        # Only process tables that look like general election matchups
        if not _is_general_election_table(df, dem_candidates, rep_candidates):
            continue

        logger.info("  Table %d: general election poll table (%d rows)", idx, len(df))

        polls = _parse_poll_table(df, race_label, "wikipedia", dem_candidates, rep_candidates)
        all_polls.extend(polls)

    logger.info("  Wikipedia: %d general election polls for %s", len(all_polls), race_label)
    return all_polls


# ---------------------------------------------------------------------------
# 270toWin scraper
# ---------------------------------------------------------------------------
def scrape_270towin(
    race_label: str,
    url: str,
    dem_candidates: list[str],
    rep_candidates: list[str],
) -> list[dict]:
    """Scrape poll data from 270toWin."""
    logger.info("  270toWin: %s", url)
    html = fetch_html(url)
    if not html:
        return []

    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception as e:
        logger.error("  Failed to parse tables from %s: %s", url, e)
        return []

    logger.info("  Found %d tables total", len(tables))

    all_polls = []
    for idx, df in enumerate(tables):
        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(str(x) for x in col if "Unnamed" not in str(x)).strip()
                for col in df.columns
            ]

        if not _has_pollster_column(df):
            continue

        # Only process general election tables
        if not _is_general_election_table(df, dem_candidates, rep_candidates):
            logger.debug("  Table %d: skipping (not general election)", idx)
            continue

        logger.info("  Table %d: general election poll table (%d rows)", idx, len(df))

        polls = _parse_poll_table(df, race_label, "270towin", dem_candidates, rep_candidates)
        all_polls.extend(polls)

    logger.info("  270toWin: %d general election polls for %s", len(all_polls), race_label)
    return all_polls


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def dedup_key(poll: dict) -> tuple:
    """Generate deduplication key: (normalized_pollster, date, race)."""
    return (poll["pollster"].lower(), poll.get("date", ""), poll["race"])


def deduplicate(polls: list[dict]) -> list[dict]:
    """Merge Wikipedia and 270toWin polls, preferring 270toWin for duplicates."""
    seen: dict[tuple, dict] = {}
    # Process 270toWin first so they win on conflicts
    ttw_polls = [p for p in polls if p.get("source") == "270towin"]
    wiki_polls = [p for p in polls if p.get("source") == "wikipedia"]

    for p in ttw_polls:
        key = dedup_key(p)
        seen[key] = p

    for p in wiki_polls:
        key = dedup_key(p)
        if key not in seen:
            seen[key] = p

    result = list(seen.values())
    n_deduped = len(polls) - len(result)
    if n_deduped > 0:
        logger.info("Deduplication removed %d duplicate polls", n_deduped)
    return result


# ---------------------------------------------------------------------------
# Build output DataFrame
# ---------------------------------------------------------------------------
def build_output_df(polls: list[dict]) -> pd.DataFrame:
    """Convert poll dicts to the output CSV schema."""
    rows = []
    for p in polls:
        state = ""
        for race_label, cfg in RACE_CONFIG.items():
            if p["race"] == race_label:
                state = cfg["state"]
                break

        notes_parts = [f"D={p['dem_pct']:.1f}% R={p['rep_pct']:.1f}%"]
        if p.get("sample_type"):
            notes_parts.append(p["sample_type"])
        notes_parts.append(f"src={p.get('source', 'unknown')}")

        rows.append(
            {
                "race": p["race"],
                "geography": state,
                "geo_level": "state",
                "dem_share": p["dem_share"],
                "n_sample": p.get("n_sample", ""),
                "date": p.get("date", ""),
                "pollster": p["pollster"],
                "notes": "; ".join(notes_parts),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                "race",
                "geography",
                "geo_level",
                "dem_share",
                "n_sample",
                "date",
                "pollster",
                "notes",
            ]
        )
    # Sort by race then date
    df = df.sort_values(["race", "date"], ascending=[True, True]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scrape 2026 election polls")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to CSV",
    )
    parser.add_argument(
        "--races",
        type=str,
        default=None,
        help="Comma-separated race filter (e.g. 'FL Governor,GA Senate')",
    )
    args = parser.parse_args()

    # Filter races if specified
    races = RACE_CONFIG
    if args.races:
        race_filter = [f"2026 {r.strip()}" for r in args.races.split(",")]
        races = {k: v for k, v in RACE_CONFIG.items() if k in race_filter}
        if not races:
            logger.error("No matching races found. Available: %s", list(RACE_CONFIG.keys()))
            return

    all_polls: list[dict] = []
    request_count = 0

    for race_label, cfg in races.items():
        logger.info("--- Scraping %s ---", race_label)
        dem_cands = cfg.get("dem_candidates", [])
        rep_cands = cfg.get("rep_candidates", [])

        # Wikipedia
        if request_count > 0:
            time.sleep(REQUEST_DELAY)
        wiki_polls = scrape_wikipedia(race_label, cfg["wiki_url"], dem_cands, rep_cands)
        all_polls.extend(wiki_polls)
        request_count += 1

        # 270toWin
        time.sleep(REQUEST_DELAY)
        ttw_polls = scrape_270towin(race_label, cfg["ttw_url"], dem_cands, rep_cands)
        all_polls.extend(ttw_polls)
        request_count += 1

    logger.info("=== Raw total: %d polls from all sources ===", len(all_polls))

    # Deduplicate
    deduped = deduplicate(all_polls)
    logger.info("=== After dedup: %d polls ===", len(deduped))

    # Build output
    df = build_output_df(deduped)

    # Report per-race counts
    logger.info("=== Per-race poll counts ===")
    for race_label in races:
        count = len(df[df["race"] == race_label])
        logger.info("  %s: %d polls", race_label, count)
    logger.info("  TOTAL: %d polls", len(df))

    if args.dry_run:
        logger.info("=== DRY RUN -- printing output ===")
        print(df.to_csv(index=False))
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info("Written %d polls to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
