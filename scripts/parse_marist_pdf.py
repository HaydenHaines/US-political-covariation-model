"""
Parse Marist Poll "NOS-and-Tables" crosstab PDFs to extract per-group vote shares
and sample composition demographics.

Extracts two types of data from Marist PDFs:

1. **Crosstab vote shares** (xt_vote_* columns): Per-demographic two-party Democratic
   share for the target question, parsed from the full crosstab table (e.g. page 13
   for NYGOV26).  Enables Tier 2 W vector construction with both a different W and a
   different y per demographic group.

2. **Sample composition** (xt_* columns): Fraction of the poll sample in each
   demographic group, parsed from the "Nature of the Sample" table on page 2.
   These are required so ``_extract_crosstabs_from_xt()`` in forecast_engine can
   build proper Tier 2 observations.  Without them the poll falls through to Tier 3.

Usage:
    uv run python scripts/parse_marist_pdf.py data/raw/marist/NYS_202602201349.pdf NYGOV26
    uv run python scripts/parse_marist_pdf.py data/raw/marist/NYS_202602201349.pdf NYGOV26 --update
    uv run python scripts/parse_marist_pdf.py data/raw/marist/NYS_202602201349.pdf NYGOV26 --update --update-composition
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"

# Mapping from Marist PDF demographic labels to our xt_vote_* column names.
# Keys are (section_header, row_label) tuples for unambiguous matching.
# We prefer the more granular Race/Ethnicity breakdown (White/Black/Latino)
# over the coarse one (White/Non-white).
DEMOGRAPHIC_MAP: dict[tuple[str, str], str] = {
    ("Education", "College graduate"): "xt_vote_education_college",
    ("Education", "Not college graduate"): "xt_vote_education_noncollege",
    ("Age", "60 or older"): "xt_vote_age_senior",
}

# Race/Ethnicity appears twice in the PDF — once coarse (White/Non-white),
# once granular (White/Black/Latino). We want the granular version.
# The granular block is the SECOND occurrence of "Race/Ethnicity".
RACE_LABELS: dict[str, str] = {
    "White": "xt_vote_race_white",
    "Black": "xt_vote_race_black",
    "Latino": "xt_vote_race_hispanic",
}


# Page 2 "Nature of the Sample" label → xt_* column names.
# We use the *Registered Voters* column because the governor question targets RVs.
# Keys are the exact text labels that appear in the left column of the NOS table.
SAMPLE_COMPOSITION_MAP: dict[str, str] = {
    "White": "xt_race_white",
    "Black": "xt_race_black",
    "Latino": "xt_race_hispanic",
    "College graduate": "xt_education_college",
    "Not college graduate": "xt_education_noncollege",
    "60 or older": "xt_age_senior",
}

# NOS table row labels that are section headers (they never carry percentages).
# We skip them while walking through the NOS text.
NOS_SECTION_HEADERS = {
    "NYS Adults",
    "NY Registered Voters",
    "Party Registration",
    "Region",
    "Gender",
    "Age",
    "Race/Ethnicity",
    "Household Income",
    "Education",
    "Education by Race",
    "Area Description",
    "Area Description - Gender",
}


def two_party_dem_share(dem_pct: float, rep_pct: float) -> Optional[float]:
    """Convert raw percentages to two-party Democratic share.

    Args:
        dem_pct: Democratic candidate percentage (0-100 scale).
        rep_pct: Republican candidate percentage (0-100 scale).

    Returns:
        Two-party share as a float in [0, 1], or None if both are zero.
    """
    if dem_pct + rep_pct == 0:
        return None
    return dem_pct / (dem_pct + rep_pct)


def find_question_pages(pdf: pdfplumber.PDF, question_code: str) -> list[int]:
    """Find 0-indexed page numbers containing the given question code.

    The question code appears at the start of a line followed by a period,
    e.g. "NYGOV26. Marist Poll New York State Tables..."
    """
    pages = []
    pattern = re.compile(rf"^{re.escape(question_code)}\.", re.MULTILINE)
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and pattern.search(text):
            pages.append(i)
    return pages


def parse_crosstab_text(text: str) -> dict[str, Optional[float]]:
    """Parse a single question's crosstab text into xt_vote_* values.

    The text has this structure:
      - A header row with candidate names
      - "Row % Row % Row % Row %" column header
      - Data rows: either "SectionHeader GroupLabel XX% YY% ..." or
        just "GroupLabel XX% YY% ..." (continuation of previous section)

    The Democrat column is always first, Republican second.

    Returns:
        Dict mapping xt_vote_* column names to two-party dem share values.
    """
    result: dict[str, Optional[float]] = {}

    # Split into lines and find the data rows (lines ending with percent patterns).
    # A data row looks like: "Some Label 50% 33% 2% 15%"
    pct_pattern = re.compile(r"(\d+)%")
    # Match lines that have 3+ percentage values (the crosstab columns).
    data_line_pattern = re.compile(
        r"^(.+?)\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s*$"
    )

    lines = text.split("\n")
    current_section = ""
    race_ethnicity_count = 0  # Track which Race/Ethnicity block we're in.

    # Known section headers that appear as prefixes on data lines.
    # When a line starts with one of these, the rest is the group label.
    section_headers = [
        "Party Registration",
        "Region",
        "Household Income",
        "Education",
        "Race/Ethnicity",
        "Race and Education",
        "Age",
        "Gender",
        "2024 Support",
        "Area Description",
    ]

    for line in lines:
        match = data_line_pattern.match(line.strip())
        if not match:
            continue

        label_part = match.group(1).strip()
        dem_pct = int(match.group(2))
        rep_pct = int(match.group(3))

        # Check if line starts with a section header.
        found_section = False
        for header in section_headers:
            if label_part.startswith(header):
                # Special handling: "Race/Ethnicity" appears twice.
                if header == "Race/Ethnicity":
                    race_ethnicity_count += 1
                current_section = header
                # The group label is the remainder after the header.
                label_part = label_part[len(header) :].strip()
                found_section = True
                break

        if not label_part:
            # This shouldn't happen — the section header IS the only text.
            continue

        group_label = label_part
        dem_share = two_party_dem_share(dem_pct, rep_pct)

        # Check for topline row (overall registered voters).
        if "Registered Voters" in group_label and current_section == "":
            result["dem_share_topline"] = dem_share
            continue

        # Map to xt_vote_* columns.
        key = (current_section, group_label)
        if key in DEMOGRAPHIC_MAP:
            result[DEMOGRAPHIC_MAP[key]] = dem_share

        # Handle Race/Ethnicity — only use the second (granular) occurrence.
        if current_section == "Race/Ethnicity" and race_ethnicity_count == 2:
            if group_label in RACE_LABELS:
                result[RACE_LABELS[group_label]] = dem_share

    return result


def parse_sample_composition(text: str) -> dict[str, float]:
    """Parse the "Nature of the Sample" page and return xt_* composition fractions.

    The NOS table has two data columns: "NYS Adults" and "NYS Registered Voters".
    We target the *Registered Voters* column because the NY Governor question
    (NYGOV26) is asked of registered voters.

    The table uses a two-column layout where each row ends with two percentages,
    e.g.::

        Race/Ethnicity White 57% 63%
        Black 14% 13%

    The first percentage is NYS Adults; the second is NYS Registered Voters.
    We extract the second value for each row that appears in SAMPLE_COMPOSITION_MAP.

    Args:
        text: Raw text extracted from the NOS page (page 2 of the Marist PDF).

    Returns:
        Dict mapping xt_* column names to fraction values in [0, 1].
        Only keys with recognized labels are included; empty dict if nothing found.
    """
    # A NOS data row has exactly two percentage values at the end.
    # We match lines ending with "NN% NN%" (with optional trailing whitespace).
    two_pct_pattern = re.compile(r"^(.+?)\s+(\d+)%\s+(\d+)%\s*$")

    result: dict[str, float] = {}
    for line in text.split("\n"):
        match = two_pct_pattern.match(line.strip())
        if not match:
            continue
        label_raw = match.group(1).strip()
        # The second percentage is the Registered Voters column.
        reg_voter_pct = int(match.group(3))

        # Strip any section header prefix (e.g. "Race/Ethnicity White" → "White").
        # Section headers are known strings; we check longest-first to avoid
        # partial matches (e.g. "Education" vs "Education by Race").
        label = label_raw
        for header in sorted(NOS_SECTION_HEADERS, key=len, reverse=True):
            if label_raw.startswith(header):
                label = label_raw[len(header):].strip()
                break

        if label in SAMPLE_COMPOSITION_MAP:
            col = SAMPLE_COMPOSITION_MAP[label]
            result[col] = reg_voter_pct / 100.0

    return result


def parse_marist_pdf(
    pdf_path: str | Path, question_code: str
) -> dict[str, Optional[float]]:
    """Parse a Marist PDF and extract crosstab vote shares for a question.

    Args:
        pdf_path: Path to the Marist NOS-and-Tables PDF.
        question_code: Question identifier (e.g., "NYGOV26").

    Returns:
        Dict mapping xt_vote_* column names to two-party dem share values.

    Raises:
        ValueError: If the question code is not found in the PDF.
    """
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        pages = find_question_pages(pdf, question_code)
        if not pages:
            raise ValueError(
                f"Question code '{question_code}' not found in {pdf_path.name}"
            )

        # Extract text from all pages containing this question.
        # Typically a single page, but handle multi-page questions.
        full_text = ""
        for page_idx in pages:
            full_text += pdf.pages[page_idx].extract_text() + "\n"

        logger.info(
            "Found %s on page(s) %s",
            question_code,
            [p + 1 for p in pages],
        )

    return parse_crosstab_text(full_text)


def extract_nos_page_text(pdf_path: str | Path, page_index: int = 1) -> str:
    """Extract raw text from the Nature of the Sample page.

    The NOS table is always on page 2 (0-indexed: 1) of Marist NOS-and-Tables PDFs.

    Args:
        pdf_path: Path to the Marist PDF.
        page_index: 0-indexed page number of the NOS table (default 1 = page 2).

    Returns:
        Raw text string extracted by pdfplumber.
    """
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        if page_index >= len(pdf.pages):
            raise ValueError(
                f"PDF only has {len(pdf.pages)} pages; "
                f"requested NOS page index {page_index}"
            )
        text = pdf.pages[page_index].extract_text() or ""
    return text


def parse_marist_pdf_composition(
    pdf_path: str | Path, nos_page_index: int = 1
) -> dict[str, float]:
    """Parse the Nature of the Sample table and return xt_* composition fractions.

    Convenience wrapper that opens the PDF, extracts the NOS page text, and
    delegates to ``parse_sample_composition()``.

    Args:
        pdf_path: Path to the Marist NOS-and-Tables PDF.
        nos_page_index: 0-indexed page number of the NOS table (default 1 = page 2).

    Returns:
        Dict mapping xt_* column names to fraction values in [0, 1].
    """
    text = extract_nos_page_text(pdf_path, nos_page_index)
    result = parse_sample_composition(text)
    logger.info(
        "Parsed NOS page (page %d): found %d composition values",
        nos_page_index + 1,
        len(result),
    )
    return result


def update_polls_csv(
    extracted: dict[str, Optional[float]],
    race_filter: Optional[str] = None,
    pollster_filter: str = "Marist",
    include_composition: bool = False,
) -> bool:
    """Update polls_2026.csv with extracted xt_vote_* and/or xt_* values.

    Matches polls by pollster name containing 'Marist'. If race_filter is
    provided, also filters by the race column.

    Args:
        extracted: Dict of column name → value pairs to write.
        race_filter: Optional substring filter on the race column.
        pollster_filter: Pollster name substring to match (default 'Marist').
        include_composition: When True, also write xt_* (sample composition)
            columns in addition to xt_vote_* columns.  When False (default),
            only vote-share columns are written — preserving backward compatibility
            with the original ``--update`` flag behaviour.

    Returns:
        True if any rows were updated, False otherwise.
    """
    if not POLLS_CSV.exists():
        logger.error("Polls CSV not found: %s", POLLS_CSV)
        return False

    # Read existing CSV.
    with open(POLLS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        logger.error("Empty CSV")
        return False

    # Decide which columns to write.
    if include_composition:
        # xt_vote_* AND xt_* (composition) columns, but not both prefixes
        # doubling up — xt_vote_* is a subset of keys that start with "xt_".
        xt_columns = [k for k in extracted if k.startswith("xt_")]
    else:
        xt_columns = [k for k in extracted if k.startswith("xt_vote_")]

    if not xt_columns:
        logger.warning("No xt_* values to update")
        return False

    updated = False
    for row in rows:
        pollster_match = pollster_filter.lower() in row.get("pollster", "").lower()
        race_match = race_filter is None or race_filter in row.get("race", "")
        if pollster_match and race_match:
            row_updated = False
            for col in xt_columns:
                if col in fieldnames and extracted[col] is not None:
                    row[col] = f"{extracted[col]:.6f}"
                    row_updated = True
            if row_updated:
                updated = True
                logger.info("Updated poll: %s / %s", row.get("race"), row.get("date"))

    if updated:
        with open(POLLS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote updated CSV to %s", POLLS_CSV)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Marist Poll crosstab PDFs for demographic vote shares "
        "and sample composition demographics."
    )
    parser.add_argument("pdf_path", help="Path to Marist NOS-and-Tables PDF")
    parser.add_argument("question_code", help="Question code (e.g., NYGOV26)")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update polls_2026.csv with extracted xt_vote_* values",
    )
    parser.add_argument(
        "--update-composition",
        action="store_true",
        help=(
            "Also parse the Nature of the Sample page and write xt_* "
            "(sample composition) columns.  Implies --update."
        ),
    )
    parser.add_argument(
        "--race",
        help="Race column filter for --update (e.g., 'NY Governor')",
    )
    args = parser.parse_args()

    # Parse crosstab vote shares.
    extracted: dict[str, Optional[float]] = parse_marist_pdf(
        args.pdf_path, args.question_code
    )

    # Parse sample composition from NOS page when requested.
    composition: dict[str, float] = {}
    if args.update_composition:
        composition = parse_marist_pdf_composition(args.pdf_path)
        # Merge into extracted so we display and write everything together.
        extracted.update(composition)

    # Display results.
    print(f"\nExtracted crosstab data for {args.question_code}:")
    print("-" * 50)
    for key, value in sorted(extracted.items()):
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: None")
    print()

    if args.update or args.update_composition:
        success = update_polls_csv(
            extracted,
            race_filter=args.race,
            include_composition=args.update_composition,
        )
        if not success:
            logger.warning("No matching polls found to update")
            sys.exit(1)


if __name__ == "__main__":
    main()
