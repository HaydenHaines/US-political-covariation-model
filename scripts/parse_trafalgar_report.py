"""
Parse Trafalgar Group report PDFs/text into WetherVane poll crosstab columns.

Trafalgar Group (TheTrafalgarGroup.com) is a multi-mode pollster that publishes
topline and crosstab reports for state and national races.  Their reports expose:

1. A "Sample Breakdown" table with composition by race, gender, age, education,
   and party.  These become xt_* columns.
2. Demographic ballot crosstabs.  These become xt_vote_* two-party Democratic
   vote-share columns.

Trafalgar-specific header format:
  "n=952 Likely General Election Voters | Survey Conducted March 18-19, 2026"

The percentage row in the sample breakdown uses a "% of Sample" prefix rather
than Cygnal's "Sample (pct)" or Quantus's "Pct".

Usage:
    uv run python scripts/parse_trafalgar_report.py report.pdf
    uv run python scripts/parse_trafalgar_report.py report.txt --race "2026 GA Senate" --update
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"

POLL_FIELDNAMES = {
    "race",
    "geography",
    "geo_level",
    "dem_share",
    "n_sample",
    "date",
    "pollster",
    "notes",
    "methodology",
}

PROFILE_LABEL_MAP = {
    "White": "xt_race_white",
    "Black": "xt_race_black",
    "Hispanic": "xt_race_hispanic",
    "Asian": "xt_race_asian",
    "65+": "xt_age_senior",
    "College": "xt_education_college",
    "College grad": "xt_education_college",
    "Collegegrad": "xt_education_college",
    "Non-College": "xt_education_noncollege",
    "Non-college": "xt_education_noncollege",
    "Noncollege": "xt_education_noncollege",
    "No degree": "xt_education_noncollege",
    "Nodegree": "xt_education_noncollege",
    "Urban": "xt_urbanicity_urban",
    "Rural": "xt_urbanicity_rural",
}

VOTE_LABEL_MAP = {
    "White": "xt_vote_race_white",
    "Black": "xt_vote_race_black",
    "Hispanic": "xt_vote_race_hispanic",
    "Asian": "xt_vote_race_asian",
    "65+": "xt_vote_age_senior",
    "College": "xt_vote_education_college",
    "College grad": "xt_vote_education_college",
    "Collegegrad": "xt_vote_education_college",
    "Non-College": "xt_vote_education_noncollege",
    "Non-college": "xt_vote_education_noncollege",
    "Noncollege": "xt_vote_education_noncollege",
    "No degree": "xt_vote_education_noncollege",
    "Nodegree": "xt_vote_education_noncollege",
    "Men": "xt_vote_gender_men",
    "Women": "xt_vote_gender_women",
}

KNOWN_LABELS = [
    "Non-College",
    "Non-college",
    "College grad",
    "Collegegrad",
    "No degree",
    "Nodegree",
    "Hispanic",
    "Independent",
    "Republican",
    "Democrat",
    "Suburban",
    "Female",
    "Women",
    "College",
    "Black",
    "White",
    "Asian",
    "Other",
    "18-29",
    "18-44",
    "30-44",
    "45-64",
    "65+",
    "Urban",
    "Rural",
    "Male",
    "Men",
    "Rep",
    "Dem",
    "Ind",
    "Totals",
    "Total",
    "Pct",
]

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def normalize_text(text: str) -> str:
    """Normalize PDF extraction artifacts."""
    replacements = {
        "\x00": "",
        "\x02": "-",
        "–": "-",
        "—": "-",
        "’": "'",
        " ": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[ \t]+", " ", text)


def two_party_dem_share(dem_pct: float, rep_pct: float) -> Optional[float]:
    """Convert Democratic/Republican percentages to two-party Democratic share."""
    if dem_pct + rep_pct == 0:
        return None
    return dem_pct / (dem_pct + rep_pct)


def _percentages(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", line)]


def _is_percentage_only_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    without_pcts = re.sub(r"\d+(?:\.\d+)?\s*%", "", stripped)
    return without_pcts.strip() == ""


def _split_known_labels(line: str) -> list[str]:
    """Return known demographic labels in the order they appear on a header line."""
    line = normalize_text(line)
    matches: list[tuple[int, str]] = []
    for label in KNOWN_LABELS:
        pattern = re.compile(rf"(?<![A-Za-z0-9\-]){re.escape(label)}(?![A-Za-z0-9\+\-])")
        for match in pattern.finditer(line):
            matches.append((match.start(), label))
    matches.sort(key=lambda item: item[0])

    labels: list[str] = []
    last_start = -1
    for start, label in matches:
        if start == last_start:
            continue
        labels.append(label)
        last_start = start
    return labels


def _align_labels_and_values(labels: list[str], values: list[float]) -> list[tuple[str, float]]:
    if labels and labels[0] in {"Totals", "Total", "Pct"} and len(labels) == len(values) + 1:
        labels = labels[1:]
    if len(labels) != len(values):
        return []
    return list(zip(labels, values))


def parse_header(text: str) -> dict[str, str | int]:
    """Extract field dates and sample size from Trafalgar methodology text.

    Trafalgar format: "n=952 Likely Voters | Survey Conducted March 18-19, 2026"
    """
    text = normalize_text(text)
    result: dict[str, str | int] = {}

    # Trafalgar uses lowercase "n=" for sample size
    sample_match = re.search(
        r"\bn=([\d,]+)\s+([^|\n]+)",
        text,
        re.IGNORECASE,
    )
    if sample_match:
        result["n_sample"] = int(sample_match.group(1).replace(",", ""))
        sample_type = sample_match.group(2).strip()
        if sample_type:
            result["notes"] = sample_type

    # Trafalgar uses "Survey Conducted Month D-D, YYYY"
    conducted_match = re.search(
        r"Survey Conducted\s+([A-Za-z]+)\s+(\d{1,2})\s*-\s*(?:([A-Za-z]+)\s+)?(\d{1,2})(?:,\s*(\d{4}))?",
        text,
        re.IGNORECASE,
    )
    if conducted_match:
        month1_name = conducted_match.group(1).lower()
        day1 = int(conducted_match.group(2))
        month2_name = (conducted_match.group(3) or conducted_match.group(1)).lower()
        day2 = int(conducted_match.group(4))
        year = int(conducted_match.group(5) or _infer_report_year(text))
        month1 = MONTH_MAP.get(month1_name)
        month2 = MONTH_MAP.get(month2_name)
        if month1 and month2:
            result["date_start"] = f"{year}-{month1:02d}-{day1:02d}"
            result["date_end"] = f"{year}-{month2:02d}-{day2:02d}"

    return result


def _infer_report_year(text: str) -> int:
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        return int(year_match.group(1))
    raise ValueError("Could not infer report year from Trafalgar text")


def parse_sample_composition(text: str) -> dict[str, float]:
    """Parse Trafalgar sample breakdown into xt_* fraction columns.

    Trafalgar sample tables appear as a single-row header + percentage row prefixed
    with "% of Sample":
        Totals  Men  Women  White  Black  Hispanic  ...  College  Non-College  ...
        % of Sample  100%  49%  51%   70%    12%     ...   42%       58%       ...
    """
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    result: dict[str, float] = {}

    for i, line in enumerate(lines[:-1]):
        labels = _split_known_labels(line)
        if len(labels) < 2:
            continue

        next_line = lines[i + 1]
        values = _percentages(next_line)
        if not values:
            continue

        nl_lower = next_line.lower()
        # Accept: "% of Sample ...", "Sample (pct) ...", "Pct ...", bare pct rows
        bare = re.sub(
            r"(?:%\s*of\s*sample|Sample\s*\(pct\)|Pct|Totals?|100%)",
            "",
            next_line,
            flags=re.IGNORECASE,
        ).strip()
        bare_without_pcts = re.sub(r"\d+(?:\.\d+)?\s*%", "", bare).strip()
        if (
            bare_without_pcts
            and not nl_lower.startswith("pct")
            and not nl_lower.startswith("sample")
            and not nl_lower.startswith("% of")
        ):
            continue

        for label, pct in _align_labels_and_values(labels, values):
            col = PROFILE_LABEL_MAP.get(label)
            if col:
                result[col] = pct / 100.0

    return result


def _classify_response_line(line: str) -> Optional[str]:
    """Return 'dem' or 'rep' if the line starts with a Democratic or Republican label."""
    prefix = re.split(r"\d+(?:\.\d+)?\s*%", line.strip(), maxsplit=1)[0].lower()
    prefix = re.sub(r"[^a-z]+", "", prefix)
    if not prefix:
        return None
    if "democratic" in prefix or prefix.startswith("democrat") or prefix == "dem":
        return "dem"
    if "republican" in prefix or prefix.startswith("republican") or prefix == "rep":
        return "rep"
    return None


def parse_demographic_vote_shares(text: str) -> dict[str, float]:
    """Parse Trafalgar demographic ballot blocks into xt_vote_* columns.

    Trafalgar crosstabs appear as section blocks:
        By Race
              White  Black  Hispanic  Asian
        Democrat       37%    78%     50%      44%
        Republican     56%    12%     36%      42%
    """
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    result: dict[str, float] = {}

    for i, line in enumerate(lines):
        labels = _split_known_labels(line)
        if len(labels) < 2:
            continue
        mapped_labels = [label for label in labels if label in VOTE_LABEL_MAP]
        if not mapped_labels:
            continue

        dem_values: list[float] | None = None
        rep_values: list[float] | None = None

        for lookahead in range(i + 1, min(i + 6, len(lines))):
            row_type = _classify_response_line(lines[lookahead])
            values = _percentages(lines[lookahead])
            if not values:
                continue
            if row_type == "dem":
                dem_values = values
            elif row_type == "rep":
                rep_values = values
            if dem_values is not None and rep_values is not None:
                break

        if dem_values is None or rep_values is None:
            pct_rows = [
                _percentages(candidate)
                for candidate in lines[i + 1 : min(i + 4, len(lines))]
                if _is_percentage_only_line(candidate)
            ]
            if len(pct_rows) >= 2:
                dem_values, rep_values = pct_rows[0], pct_rows[1]

        if dem_values is None or rep_values is None:
            continue

        aligned_dem = _align_labels_and_values(labels, dem_values)
        aligned_rep = _align_labels_and_values(labels, rep_values)
        if not aligned_dem or not aligned_rep:
            continue

        dem_by_label = dict(aligned_dem)
        rep_by_label = dict(aligned_rep)
        for label in labels:
            col = VOTE_LABEL_MAP.get(label)
            if not col or label not in dem_by_label or label not in rep_by_label:
                continue
            share = two_party_dem_share(dem_by_label[label], rep_by_label[label])
            if share is not None:
                result[col] = share

    return result


def parse_trafalgar_text(text: str) -> dict[str, object]:
    """Parse all supported Trafalgar fields from extracted report text."""
    normalized = normalize_text(text)
    parsed: dict[str, object] = {
        "pollster": "Trafalgar Group",
        "methodology": "Multi-Mode",
    }
    parsed.update(parse_header(normalized))
    parsed.update(parse_sample_composition(normalized))
    parsed.update(parse_demographic_vote_shares(normalized))
    return parsed


def extract_text(path: str | Path) -> str:
    """Extract text from a Trafalgar PDF or read an already-extracted text file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        if not _HAS_PDFPLUMBER:
            raise ImportError("pdfplumber is required for PDF extraction: pip install pdfplumber")
        import pdfplumber as _pdfplumber
        with _pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text()


def parse_trafalgar_report(path: str | Path) -> dict[str, object]:
    """Parse a Trafalgar report PDF/text file into polls_2026.csv-compatible fields."""
    return parse_trafalgar_text(extract_text(path))


def update_polls_csv(
    extracted: dict[str, object],
    race_filter: Optional[str] = None,
    pollster_filter: str = "Trafalgar",
) -> bool:
    """Update matching Trafalgar rows in polls_2026.csv with parsed xt_* fields."""
    if not POLLS_CSV.exists():
        logger.error("Polls CSV not found: %s", POLLS_CSV)
        return False

    with open(POLLS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        logger.error("Empty CSV")
        return False

    update_columns = [
        key
        for key in extracted
        if key in fieldnames and (key.startswith("xt_") or key in POLL_FIELDNAMES)
    ]
    if not update_columns:
        logger.warning("No polls_2026.csv-compatible fields parsed")
        return False

    updated = False
    for row in rows:
        pollster_match = pollster_filter.lower() in row.get("pollster", "").lower()
        race_match = race_filter is None or race_filter in row.get("race", "")
        if not pollster_match or not race_match:
            continue
        for col in update_columns:
            value = extracted[col]
            if value is None:
                continue
            row[col] = f"{value:.6f}" if isinstance(value, float) else str(value)
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
        description="Parse Trafalgar Group poll reports for xt_* poll crosstab fields."
    )
    parser.add_argument("path", help="Path to a Trafalgar report PDF or extracted text")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update polls_2026.csv with parsed xt_* fields",
    )
    parser.add_argument("--race", help="Race column filter for --update")
    args = parser.parse_args()

    extracted = parse_trafalgar_report(args.path)

    print("\nExtracted Trafalgar crosstab data:")
    print("-" * 50)
    for key, value in sorted(extracted.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    if args.update:
        success = update_polls_csv(extracted, race_filter=args.race)
        if not success:
            logger.warning("No matching polls found to update")
            sys.exit(1)


if __name__ == "__main__":
    main()
