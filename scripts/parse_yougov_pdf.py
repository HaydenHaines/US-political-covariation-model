"""
Parse YouGov/Economist weekly tracker PDFs to extract generic ballot vote shares
by demographic group.

Extracts column-oriented crosstab data from the GenericCongressionalVote question
and converts to two-party Democratic share for use in the WetherVane poll pipeline.

Usage:
    uv run python scripts/parse_yougov_pdf.py data/raw/yougov/econTabReport_o84FoNw.pdf
    uv run python scripts/parse_yougov_pdf.py --download-all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "yougov"
POLLS_DIR = PROJECT_ROOT / "data" / "polls"

# Known 2026 PDF hashes. Maps (start_date, end_date) to CloudFront hash.
KNOWN_PDFS_2026: dict[tuple[str, str], str] = {
    ("2026-01-02", "2026-01-05"): "aY8mpiN",
    ("2026-01-09", "2026-01-12"): "qsNv5iE",
    ("2026-01-16", "2026-01-19"): "z9wtNZI",
    ("2026-01-23", "2026-01-26"): "8FWGyNz",
    ("2026-02-06", "2026-02-09"): "vNnwPx2",
    ("2026-03-06", "2026-03-09"): "EcCnfRV",
    ("2026-03-13", "2026-03-16"): "CwWXhS2",
    ("2026-03-20", "2026-03-23"): "o84FoNw",
}

BASE_URL = "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_{hash}.pdf"

# The first table has these demographic columns in order. The column headers are
# concatenated without spaces in the PDF (e.g., "Nodegree", "Collegegrad").
# We map them to our gb_vote_* names.
TABLE1_COLUMNS = [
    "Total",
    "Male",
    "Female",
    "White",
    "Black",
    "Hispanic",
    "18-29",
    "30-44",
    "45-64",
    "65+",
    "Nodegree",
    "Collegegrad",
]

TABLE1_COLUMN_MAP = {
    "Total": "gb_topline",
    "Male": "gb_vote_gender_male",
    "Female": "gb_vote_gender_female",
    "White": "gb_vote_race_white",
    "Black": "gb_vote_race_black",
    "Hispanic": "gb_vote_race_hispanic",
    "18-29": "gb_vote_age_young",
    "65+": "gb_vote_age_senior",
    "Nodegree": "gb_vote_education_noncollege",
    "Collegegrad": "gb_vote_education_college",
}

# Second table columns (for party ID breakdown, less critical but nice to have).
TABLE2_COLUMNS = [
    "Total",
    "Harris",
    "Trump",
    "Voters",
    "Lib",
    "Mod",
    "Con",
    "Supporter",
    "Dem",
    "Ind",
    "Rep",
]

TABLE2_COLUMN_MAP = {
    "Dem": "gb_vote_party_dem",
    "Ind": "gb_vote_party_ind",
    "Rep": "gb_vote_party_rep",
}

# Month name to number for date parsing.
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
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


def parse_header(pdf: pdfplumber.PDF) -> dict[str, str | int]:
    """Extract date range and sample size from the PDF header.

    The header on every page follows the pattern:
        "The Economist/YouGov Poll"
        "March 20 - 23, 2026 - 1665 U.S. Adult Citizens"

    Returns:
        Dict with date_start, date_end (YYYY-MM-DD), n_total.
    """
    # Check first few pages for the header — it's on every page.
    for page in pdf.pages[:3]:
        text = page.extract_text() or ""
        # Pattern: "Month DD - DD, YYYY - NNNN U.S. Adult Citizens"
        match = re.search(
            r"(\w+)\s+(\d+)\s*-\s*(\d+),\s*(\d{4})\s*-\s*(\d[\d,]*)\s*U\.?S\.?\s*Adult",
            text,
        )
        if match:
            month_name = match.group(1).lower()
            day_start = int(match.group(2))
            day_end = int(match.group(3))
            year = int(match.group(4))
            n_total = int(match.group(5).replace(",", ""))
            month = MONTH_MAP.get(month_name)
            if month is None:
                continue
            return {
                "date_start": f"{year}-{month:02d}-{day_start:02d}",
                "date_end": f"{year}-{month:02d}-{day_end:02d}",
                "n_total": n_total,
            }
    raise ValueError("Could not parse date/sample from PDF header")


def find_generic_ballot_page(pdf: pdfplumber.PDF) -> Optional[int]:
    """Find the 0-indexed page with the GenericCongressionalVote question.

    Searches for the question header pattern: "NN. GenericCongressionalVote"
    or variations like "GenericCongress" appearing after a number+period.
    """
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        # The question number varies across issues, so match any number prefix.
        # Require percentage data on the page too — the table of contents also
        # mentions the question name but has no data rows.
        if re.search(r"\d+\.\s*GenericCongressionalVote", text) and re.search(
            r"\d+%\s+\d+%\s+\d+%", text
        ):
            return i
    return None


def parse_percentage_row(line: str, n_columns: int) -> Optional[list[int]]:
    """Extract percentage values from a crosstab row.

    YouGov rows look like: "TheDemocraticPartycandidate 39% 33% 44% 34% 65% ..."
    or simply: "Dem 39% 33% ..."

    Returns:
        List of integer percentages, or None if the line doesn't match.
    """
    # Find all NN% values in the line.
    pcts = re.findall(r"(\d+)%", line)
    if len(pcts) < n_columns:
        return None
    # Take only the first n_columns values (in case of trailing noise).
    return [int(p) for p in pcts[:n_columns]]


def parse_unweighted_n(line: str, n_columns: int) -> Optional[list[int]]:
    """Extract unweighted N values from the UnweightedN row.

    Format: "UnweightedN (1,664) (784) (880) (1,096) ..."

    Returns:
        List of integer sample sizes, or None if not an N row.
    """
    if "UnweightedN" not in line and "Unweighted N" not in line:
        return None
    # Match parenthesized numbers, with or without commas.
    ns = re.findall(r"\(([\d,]+)\)", line)
    if len(ns) < n_columns:
        return None
    return [int(n.replace(",", "")) for n in ns[:n_columns]]


def classify_response_row(line: str) -> Optional[str]:
    """Classify a response row as 'dem', 'rep', 'other', 'notsure', 'notvote', or 'totals'.

    YouGov uses various phrasings:
    - "TheDemocraticPartycandidate" or just "Dem" or "Democrat"
    - "TheRepublicanPartycandidate" or just "Rep" or "Republican"
    - "Other"
    - "Notsure"
    - "Iwouldnotvote" or "Wouldnotvote"
    - "Totals"
    """
    # Normalize: strip, lowercase, remove spaces for matching.
    normalized = line.strip().lower().replace(" ", "")
    # Check prefix before the first percentage.
    prefix = re.split(r"\d+%", normalized)[0].strip()

    if not prefix:
        return None

    if "democratic" in prefix or prefix.startswith("dem"):
        return "dem"
    if "republican" in prefix or prefix.startswith("rep"):
        return "rep"
    if prefix.startswith("other"):
        return "other"
    if "notsure" in prefix:
        return "notsure"
    if "wouldnotvote" in prefix or "iwouldnotvote" in prefix:
        return "notvote"
    if prefix.startswith("total"):
        return "totals"
    return None


def parse_generic_ballot_page(text: str) -> dict[str, list[int]]:
    """Parse the generic ballot page text into dem/rep percentage arrays.

    The page contains two tables (table 1: demographics, table 2: political groups).
    Each table has the same response rows (Dem, Rep, Other, Notsure, etc.).

    Returns:
        Dict with keys: 'dem_t1', 'rep_t1', 'n_t1', 'dem_t2', 'rep_t2', 'n_t2'
        where t1/t2 are table 1 and table 2. Values are lists of integers
        corresponding to the column order defined above.
    """
    lines = text.split("\n")
    result: dict[str, list[int]] = {}

    # We need to identify two tables on the page. The first table starts after
    # the question text and has 12 columns (Sex+Race+Age+Education). The second
    # table has 11 columns (2024Vote+Reg+Ideology+MAGA+PartyID).
    #
    # Strategy: find the Dem row(s) by classification. The first Dem row with
    # 12 percentage values is table 1; the next one with 11 is table 2.

    table1_found = False
    table2_found = False

    for line in lines:
        row_type = classify_response_row(line)

        if row_type == "dem":
            # Try table 1 first (12 columns).
            if not table1_found:
                pcts = parse_percentage_row(line, len(TABLE1_COLUMNS))
                if pcts:
                    result["dem_t1"] = pcts
                    table1_found = True
                    continue
            # Then table 2 (11 columns).
            if table1_found and not table2_found:
                pcts = parse_percentage_row(line, len(TABLE2_COLUMNS))
                if pcts:
                    result["dem_t2"] = pcts
                    table2_found = True
                    continue

        elif row_type == "rep":
            if "dem_t1" in result and "rep_t1" not in result:
                pcts = parse_percentage_row(line, len(TABLE1_COLUMNS))
                if pcts:
                    result["rep_t1"] = pcts
            elif "dem_t2" in result and "rep_t2" not in result:
                pcts = parse_percentage_row(line, len(TABLE2_COLUMNS))
                if pcts:
                    result["rep_t2"] = pcts

        # Unweighted N rows.
        if "UnweightedN" in line or "Unweighted N" in line:
            if "n_t1" not in result:
                ns = parse_unweighted_n(line, len(TABLE1_COLUMNS))
                if ns:
                    result["n_t1"] = ns
            elif "n_t2" not in result:
                ns = parse_unweighted_n(line, len(TABLE2_COLUMNS))
                if ns:
                    result["n_t2"] = ns

    return result


def parse_yougov_pdf(pdf_path: str | Path) -> dict[str, object]:
    """Parse a YouGov/Economist PDF and extract generic ballot crosstabs.

    Args:
        pdf_path: Path to the econTabReport PDF.

    Returns:
        Dict with date_start, date_end, n_total, gb_topline, and gb_vote_* values.

    Raises:
        ValueError: If the generic ballot question is not found in the PDF.
    """
    pdf_path = Path(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        header = parse_header(pdf)
        page_idx = find_generic_ballot_page(pdf)
        if page_idx is None:
            raise ValueError(
                f"GenericCongressionalVote question not found in {pdf_path.name}"
            )

        # The question may span the current page and the next (two tables).
        page_text = pdf.pages[page_idx].extract_text() or ""
        # Check if second table is on the next page.
        if page_idx + 1 < len(pdf.pages):
            next_text = pdf.pages[page_idx + 1].extract_text() or ""
            # Only append if the next page continues the same question
            # (doesn't start a new numbered question as its first content).
            if not re.match(r".*\n\d+\.\s*[A-Z]", next_text[:200]):
                page_text += "\n" + next_text

        logger.info(
            "Found GenericCongressionalVote on page %d of %s",
            page_idx + 1,
            pdf_path.name,
        )

    raw = parse_generic_ballot_page(page_text)

    if "dem_t1" not in raw or "rep_t1" not in raw:
        raise ValueError(
            f"Could not parse Dem/Rep rows from generic ballot in {pdf_path.name}"
        )

    result: dict[str, object] = {
        "date_start": header["date_start"],
        "date_end": header["date_end"],
        "n_total": header["n_total"],
    }

    # Convert table 1 columns to two-party dem shares.
    for i, col_name in enumerate(TABLE1_COLUMNS):
        if col_name in TABLE1_COLUMN_MAP:
            field = TABLE1_COLUMN_MAP[col_name]
            dem_pct = raw["dem_t1"][i]
            rep_pct = raw["rep_t1"][i]
            share = two_party_dem_share(dem_pct, rep_pct)
            if share is not None:
                result[field] = round(share, 4)

    # Extract sample sizes from table 1 UnweightedN.
    if "n_t1" in raw:
        n_map = {
            "Total": "n_total_ballot",
            "Male": "n_male",
            "Female": "n_female",
            "White": "n_white",
            "Black": "n_black",
            "Hispanic": "n_hispanic",
            "18-29": "n_age_young",
            "65+": "n_age_senior",
            "Nodegree": "n_education_noncollege",
            "Collegegrad": "n_education_college",
        }
        for i, col_name in enumerate(TABLE1_COLUMNS):
            if col_name in n_map:
                result[n_map[col_name]] = raw["n_t1"][i]

    # Convert table 2 columns (party ID) to two-party dem shares.
    if "dem_t2" in raw and "rep_t2" in raw:
        for i, col_name in enumerate(TABLE2_COLUMNS):
            if col_name in TABLE2_COLUMN_MAP:
                field = TABLE2_COLUMN_MAP[col_name]
                dem_pct = raw["dem_t2"][i]
                rep_pct = raw["rep_t2"][i]
                share = two_party_dem_share(dem_pct, rep_pct)
                if share is not None:
                    result[field] = round(share, 4)

    return result


def download_pdf(hash_code: str, dest_dir: Path) -> Path:
    """Download a YouGov PDF from CloudFront.

    Args:
        hash_code: The unique hash part of the URL (e.g., "o84FoNw").
        dest_dir: Directory to save the PDF.

    Returns:
        Path to the downloaded file.
    """
    import urllib.request

    url = BASE_URL.format(hash=hash_code)
    filename = f"econTabReport_{hash_code}.pdf"
    dest = dest_dir / filename

    if dest.exists():
        logger.info("Already downloaded: %s", filename)
        return dest

    logger.info("Downloading %s ...", url)
    urllib.request.urlretrieve(url, dest)
    logger.info("Saved to %s", dest)
    return dest


def download_and_parse_all() -> list[dict[str, object]]:
    """Download all known 2026 PDFs and parse generic ballot from each.

    Returns:
        List of parsed results, sorted by date_start.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    POLLS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for (date_start, date_end), hash_code in sorted(KNOWN_PDFS_2026.items()):
        try:
            pdf_path = download_pdf(hash_code, RAW_DIR)
            parsed = parse_yougov_pdf(pdf_path)
            results.append(parsed)
            logger.info(
                "Parsed %s: topline=%.3f, n=%s",
                parsed["date_start"],
                parsed.get("gb_topline", 0),
                parsed.get("n_total", "?"),
            )
        except Exception as e:
            logger.error("Failed to parse %s (%s): %s", date_start, hash_code, e)

    # Sort by date.
    results.sort(key=lambda r: r["date_start"])

    # Write combined JSON.
    output_path = POLLS_DIR / "yougov_generic_ballot_2026.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %d polls to %s", len(results), output_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse YouGov/Economist tracker PDFs for generic ballot crosstabs."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to a YouGov econTabReport PDF",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all known 2026 PDFs and produce combined JSON",
    )
    args = parser.parse_args()

    if args.download_all:
        results = download_and_parse_all()
        print(f"\nParsed {len(results)} YouGov polls:")
        for r in results:
            print(f"  {r['date_start']} to {r['date_end']}: "
                  f"topline={r.get('gb_topline', 'N/A')}, n={r.get('n_total', '?')}")
        return

    if not args.pdf_path:
        parser.error("Either provide a PDF path or use --download-all")

    extracted = parse_yougov_pdf(args.pdf_path)

    # Display results as JSON.
    print(json.dumps(extracted, indent=2))


if __name__ == "__main__":
    main()
