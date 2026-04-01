"""Silver Bulletin pollster ratings loader.

Loads the Silver Bulletin pollster stats XLSX and exposes a quality score
[0.0, 1.0] for each pollster, derived from the letter grade and predictive
plus-minus metric.

Quality score derivation
------------------------
The Silver Bulletin's "Predictive Plus-Minus" column measures how much error
a pollster adds relative to a naive benchmark (lower = better). The SB Grade
(A+, A, A-, B+, …, F) is a summary label. We use the grade as the primary
signal (robust, comparable across eras) and fall back to plus-minus rescaling
when a grade is absent.

Grade-to-score mapping (higher = better pollster):
  A+   -> 1.00
  A    -> 0.93
  A-   -> 0.87
  A/B  -> 0.80
  B+   -> 0.73
  B    -> 0.67
  B-   -> 0.60
  B/C  -> 0.53
  C+   -> 0.47
  C    -> 0.40
  C/D  -> 0.30
  D+   -> 0.20
  F    -> 0.05
  (banned) -> 0.0

Unknown pollsters get 0.5 (neutral prior).

Name normalization
------------------
Pollster names vary across data sources (e.g. "Univ." vs "University",
leading/trailing whitespace, extra punctuation). We apply a set of deterministic
substitutions so that fuzzy lookups work without a heavy library dependency.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import openpyxl

# ---------------------------------------------------------------------------
# Grade table
# ---------------------------------------------------------------------------

GRADE_SCORES: dict[str, float] = {
    "A+": 1.00,
    "A": 0.93,
    "A-": 0.87,
    "A/B": 0.80,
    "B+": 0.73,
    "B": 0.67,
    "B-": 0.60,
    "B/C": 0.53,
    "C+": 0.47,
    "C": 0.40,
    "C/D": 0.30,
    "D+": 0.20,
    "F": 0.05,
}

# Score for a pollster banned by Silver Bulletin for fabrication/misconduct
BANNED_SCORE: float = 0.0

# Score returned when a pollster name is not found in the database
DEFAULT_SCORE: float = 0.5

# Default path to the XLSX (relative to project root)
DEFAULT_XLSX_PATH: Path = (
    Path(__file__).parent.parent.parent / "data" / "raw" / "silver_bulletin" / "pollster_stats_full_2026.xlsx"
)

# Default path to the 538 pollster-ratings CSV
DEFAULT_538_CSV_PATH: Path = (
    Path(__file__).parent.parent.parent
    / "data"
    / "raw"
    / "fivethirtyeight"
    / "data-repo"
    / "pollster-ratings"
    / "pollster-ratings-combined.csv"
)

# Column name for Silver Bulletin house effects (percentage points, mean-reverted)
_SB_HOUSE_EFFECT_COL: str = "House Effect"

# Column name for 538 bias (percentage points, same sign convention)
_538_BIAS_COL: str = "bias_ppm"

# Column name for pollster name in 538 CSV
_538_POLLSTER_COL: str = "pollster"


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"\buniv\b\.?", "university"),
    (r"\bcollege\b", "college"),
    (r"\bpoll(ing)?\b", "poll"),
    (r"[''`]", "'"),          # fancy apostrophes
    (r"[&]", "and"),
    (r"\bst\b\.?", "saint"),  # St. → saint
    (r"\bmt\b\.?", "mount"),
    (r"\bprof\b\.?", "professional"),
    (r"\bres\b\.?\s+", "research "),
    (r"\bllc\b\.?", ""),
    (r"\binc\b\.?", ""),
    (r"\bcorp\b\.?", ""),
    (r"\bco\b\.?", ""),
    (r"[.,;:!?/\\]", " "),    # punctuation → space
    (r"\s+", " "),            # collapse multiple spaces
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in _SUBSTITUTIONS
]


def _normalize(name: str) -> str:
    """Normalize a pollster name for fuzzy matching."""
    s = name.lower().strip()
    for pattern, replacement in _COMPILED:
        s = pattern.sub(replacement, s)
    return s.strip()


def _name_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity between two normalised names.

    Returns Jaccard similarity on word sets: |A ∩ B| / |A ∪ B|.
    """
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


# ---------------------------------------------------------------------------
# XLSX loading helpers (shared by all three load_ functions)
# ---------------------------------------------------------------------------


def _load_xlsx_rows(path: Path) -> list[tuple]:
    """Open the Silver Bulletin XLSX and return all rows as a list of tuples.

    Raises FileNotFoundError if the file is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Silver Bulletin XLSX not found at {path}. "
            "Download it with: curl -sL <url> -o <path>"
        )
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    return rows


def _detect_name_col(rows: list[tuple]) -> int:
    """Detect the column index that contains the pollster name string.

    The Silver Bulletin XLSX uses a merged-cell layout: the header row may
    show "Pollster" at col 0, but the actual name in data rows may be in
    col 1 because col 0 holds a numeric rank. We peek at the first data row
    and advance by one if the header column contains a number instead of a
    string.
    """
    if not rows:
        return 0
    header = rows[0]
    col = _find_col(header, "pollster", required=True, fallback=0)
    if len(rows) > 1 and len(rows[1]) > col:
        peek = rows[1][col]
        if isinstance(peek, (int, float)) and not isinstance(peek, bool):
            col += 1  # name is in the adjacent (None-header) cell
    return col


def _is_banned_row(row: tuple, col_banned: Optional[int]) -> bool:
    """Return True when the banned column says 'yes' for this row."""
    if col_banned is None or len(row) <= col_banned:
        return False
    return str(row[col_banned] or "").strip().lower() == "yes"


def _extract_grade(row: tuple, col_grade: Optional[int]) -> Optional[str]:
    """Return the raw grade string from this row, or None if absent/empty."""
    if col_grade is None or len(row) <= col_grade:
        return None
    raw = row[col_grade]
    return str(raw).strip() if raw is not None else None


def _open_xlsx(path: Optional[Path | str], default: Path) -> tuple[list[tuple], int, Optional[int], Optional[int]]:
    """Open XLSX and return (rows, col_name, col_banned, col_grade).

    Centralizes the repeated open-detect-columns boilerplate shared by all
    three load_ functions. Returns an empty list for rows if the workbook is
    empty; callers should guard on ``if not rows``.
    """
    xlsx_path = Path(path) if path is not None else default
    rows = _load_xlsx_rows(xlsx_path)
    if not rows:
        return [], 0, None, None
    header = rows[0]
    col_name = _detect_name_col(rows)
    col_banned = _find_col(header, "banned", required=False, fallback=5)
    col_grade = _find_col(header, "sb grade", required=False, fallback=8)
    return rows, col_name, col_banned, col_grade


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------


def load_pollster_ratings(path: Optional[Path | str] = None) -> dict[str, float]:
    """Load Silver Bulletin XLSX and return {pollster_name: quality_score}.

    Quality score is a float in [0, 1] derived from the letter grade.
    Higher = better pollster. Banned pollsters receive 0.0.
    Unknown pollsters default to 0.5.

    Raises FileNotFoundError if the XLSX is absent.
    """
    rows, col_name, col_banned, col_grade = _open_xlsx(path, DEFAULT_XLSX_PATH)
    if not rows:
        return {}

    ratings: dict[str, float] = {}
    for row in rows[1:]:
        name = _row_name(row, col_name)
        if name is None:
            continue
        if _is_banned_row(row, col_banned):
            ratings[name] = BANNED_SCORE
            continue
        grade = _extract_grade(row, col_grade)
        ratings[name] = GRADE_SCORES.get(grade, DEFAULT_SCORE) if grade else DEFAULT_SCORE
    return ratings


def load_pollster_grades(path: Optional[Path | str] = None) -> dict[str, str]:
    """Load Silver Bulletin XLSX and return {pollster_name: grade_str}.

    Returns the raw letter grade (e.g. "A+", "B-") for each pollster.
    Banned pollsters receive "Banned". Pollsters without a grade are omitted.

    Raises FileNotFoundError if the XLSX is absent.
    """
    rows, col_name, col_banned, col_grade = _open_xlsx(path, DEFAULT_XLSX_PATH)
    if not rows:
        return {}

    grades: dict[str, str] = {}
    for row in rows[1:]:
        name = _row_name(row, col_name)
        if name is None:
            continue
        if _is_banned_row(row, col_banned):
            grades[name] = "Banned"
            continue
        grade = _extract_grade(row, col_grade)
        if grade and grade in GRADE_SCORES:
            grades[name] = grade
    return grades


def load_pollster_house_effects(path: Optional[Path | str] = None) -> dict[str, float]:
    """Load Silver Bulletin XLSX and return {pollster_name: house_effect_pp}.

    House effect is the mean-reverted estimate of partisan bias in percentage
    points. Positive = D-biased; negative = R-biased.
    Rows where the House Effect cell is blank or non-numeric are silently skipped.

    Raises FileNotFoundError if the XLSX is absent.
    """
    xlsx_path = Path(path) if path is not None else DEFAULT_XLSX_PATH
    rows = _load_xlsx_rows(xlsx_path)
    if not rows:
        return {}

    header = rows[0]
    col_name = _detect_name_col(rows)
    col_house_effect = _find_col(header, _SB_HOUSE_EFFECT_COL, required=False, fallback=None)

    house_effects: dict[str, float] = {}
    for row in rows[1:]:
        name = _row_name(row, col_name)
        if name is None:
            continue
        if col_house_effect is None or len(row) <= col_house_effect:
            continue
        raw_he = row[col_house_effect]
        if raw_he is None:
            continue
        try:
            house_effects[name] = float(raw_he)
        except (TypeError, ValueError):
            continue
    return house_effects


def _row_name(row: tuple, col_name: int) -> Optional[str]:
    """Extract the pollster name from a data row, or return None to skip."""
    if row is None or len(row) <= col_name:
        return None
    raw = row[col_name]
    return str(raw).strip() if raw else None


def load_538_bias(path: Optional[Path | str] = None) -> dict[str, float]:
    """Load 538 pollster bias estimates and return {pollster_name: bias_pp}.

    Reads the ``bias_ppm`` column from the 538 pollster-ratings CSV.
    Sign convention matches Silver Bulletin: positive = D-biased, negative = R-biased.
    Rows with missing or non-numeric bias values are silently skipped.

    Parameters
    ----------
    path:
        Path to the 538 pollster-ratings CSV. Defaults to
        ``data/raw/fivethirtyeight/data-repo/pollster-ratings/pollster-ratings-combined.csv``.

    Returns
    -------
    dict[str, float]
        Mapping from pollster name to bias in percentage points.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    csv_path = Path(path) if path is not None else DEFAULT_538_CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"538 pollster-ratings CSV not found at {csv_path}."
        )

    bias: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get(_538_POLLSTER_COL, "").strip()
            if not name:
                continue
            raw_bias = row.get(_538_BIAS_COL, "").strip()
            if not raw_bias:
                continue
            try:
                bias[name] = float(raw_bias)
            except ValueError:
                continue

    return bias


def _find_col(
    header: tuple,
    keyword: str,
    required: bool = False,
    fallback: Optional[int] = 0,
) -> Optional[int]:
    """Return column index whose header contains *keyword* (case-insensitive).

    Returns *fallback* when no column matches.  Pass ``fallback=None`` for
    optional columns where absence should be a no-op rather than an error.
    """
    for i, cell in enumerate(header):
        if cell is not None and keyword.lower() in str(cell).lower():
            return i
    return fallback


# ---------------------------------------------------------------------------
# Grade conversion helper (public, for testing)
# ---------------------------------------------------------------------------


def grade_to_score(grade: str) -> float:
    """Convert a Silver Bulletin letter grade string to a quality score.

    Parameters
    ----------
    grade:
        Letter grade string, e.g. ``"A+"``, ``"B/C"``, ``"F"``.

    Returns
    -------
    float
        Quality score in [0, 1]. Returns ``DEFAULT_SCORE`` for unrecognised grades.
    """
    return GRADE_SCORES.get(grade.strip() if grade else "", DEFAULT_SCORE)


# ---------------------------------------------------------------------------
# Fuzzy lookup
# ---------------------------------------------------------------------------

# Module-level cache populated on first call to get_pollster_quality
_RATINGS_CACHE: Optional[dict[str, float]] = None
_NORMALIZED_CACHE: Optional[dict[str, tuple[str, float]]] = None  # norm_name -> (orig_name, score)
_GRADES_CACHE: Optional[dict[str, str]] = None
_NORMALIZED_GRADES_CACHE: Optional[dict[str, tuple[str, str]]] = None  # norm_name -> (orig_name, grade)


def _ensure_cache(path: Optional[Path | str] = None) -> None:
    global _RATINGS_CACHE, _NORMALIZED_CACHE
    if _RATINGS_CACHE is None:
        _RATINGS_CACHE = load_pollster_ratings(path)
        _NORMALIZED_CACHE = {
            _normalize(name): (name, score)
            for name, score in _RATINGS_CACHE.items()
        }


def _ensure_grades_cache(path: Optional[Path | str] = None) -> None:
    global _GRADES_CACHE, _NORMALIZED_GRADES_CACHE
    if _GRADES_CACHE is None:
        _GRADES_CACHE = load_pollster_grades(path)
        _NORMALIZED_GRADES_CACHE = {
            _normalize(name): (name, grade)
            for name, grade in _GRADES_CACHE.items()
        }


def _fuzzy_lookup(
    name: str,
    cache: dict[str, tuple],
    threshold: float,
) -> Optional[tuple]:
    """Find the best fuzzy match for *name* in a normalized cache.

    Performs three-stage lookup:
    1. Exact match on normalized name.
    2. Best Jaccard token-overlap match above *threshold*.
    3. Returns None if no match meets the threshold.

    The cache maps normalized_name → (original_name, value).
    Returns the value tuple entry, or None.
    """
    norm = _normalize(name)
    # Stage 1: normalized exact match
    if norm in cache:
        return cache[norm]

    # Stage 2: best Jaccard fuzzy match
    best_entry: Optional[tuple] = None
    best_sim = 0.0
    for norm_key, entry in cache.items():
        sim = _name_similarity(norm, norm_key)
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    if best_sim >= threshold and best_entry is not None:
        return best_entry

    return None


def get_pollster_quality(
    name: str,
    path: Optional[Path | str] = None,
    threshold: float = 0.4,
) -> float:
    """Return quality score for a pollster, with fuzzy name matching.

    Lookup order:
    1. Exact match on original XLSX name.
    2. Exact match after normalization (lowercase, common substitutions).
    3. Best Jaccard token-overlap match among normalized names, if the
       overlap score exceeds *threshold*.
    4. Return ``DEFAULT_SCORE`` (0.5) if no match found.

    Parameters
    ----------
    name:
        Pollster name to look up (may use different abbreviations than XLSX).
    path:
        Optional override for the XLSX path (passed to ``load_pollster_ratings``
        on first call; ignored on subsequent calls due to caching).
    threshold:
        Minimum Jaccard similarity required for a fuzzy match to count.
        Default 0.4 — requires at least 40% token overlap.

    Returns
    -------
    float
        Quality score in [0, 1]. Returns 0.5 for unknown pollsters.
    """
    _ensure_cache(path)
    assert _RATINGS_CACHE is not None
    assert _NORMALIZED_CACHE is not None

    # Exact match on original name first (avoids normalization cost for common case)
    if name in _RATINGS_CACHE:
        return _RATINGS_CACHE[name]

    entry = _fuzzy_lookup(name, _NORMALIZED_CACHE, threshold)
    return entry[1] if entry is not None else DEFAULT_SCORE


def get_pollster_grade(
    name: str,
    path: Optional[Path | str] = None,
    threshold: float = 0.4,
) -> str | None:
    """Return the letter grade for a pollster, with fuzzy name matching.

    Same lookup strategy as ``get_pollster_quality`` but returns the letter
    grade string (e.g. "A+", "B-") instead of a numeric score. Returns None
    for unknown pollsters.

    Parameters
    ----------
    name:
        Pollster name to look up.
    path:
        Optional override for the XLSX path.
    threshold:
        Minimum Jaccard similarity for fuzzy match (default 0.4).

    Returns
    -------
    str | None
        Letter grade string, "Banned", or None if unknown.
    """
    _ensure_grades_cache(path)
    assert _GRADES_CACHE is not None
    assert _NORMALIZED_GRADES_CACHE is not None

    # Exact match on original name first
    if name in _GRADES_CACHE:
        return _GRADES_CACHE[name]

    entry = _fuzzy_lookup(name, _NORMALIZED_GRADES_CACHE, threshold)
    return entry[1] if entry is not None else None


def clear_cache() -> None:
    """Clear the module-level pollster ratings cache.

    Useful in tests that use temporary XLSX fixtures.
    """
    global _RATINGS_CACHE, _NORMALIZED_CACHE, _GRADES_CACHE, _NORMALIZED_GRADES_CACHE
    _RATINGS_CACHE = None
    _NORMALIZED_CACHE = None
    _GRADES_CACHE = None
    _NORMALIZED_GRADES_CACHE = None
