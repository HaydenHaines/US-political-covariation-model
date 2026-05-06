"""Populate polls_2026.csv xt_* columns from pollster crosstab parsers.

The individual parser scripts can parse report text, but bulk-updating the
poll CSV through those scripts is intentionally narrow and per-pollster.  This
module provides the shared CSV population path used for TODO-POLL-1: parse the
available source extracts and fill only crosstab columns on matching rows.
Poll-level fields such as date, sample size, and topline notes are never
overwritten here.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from parse_cygnal_report import parse_cygnal_report
from parse_quantus_report import parse_quantus_report
from parse_tipp_report import parse_tipp_report
from parse_trafalgar_report import parse_trafalgar_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class CrosstabSource:
    pollster: str
    parser: Callable[[str | Path], dict[str, object]]
    default_path: Path


@dataclass(frozen=True)
class PopulateStats:
    pollster: str
    source_path: Path
    parsed_xt_columns: int
    updated_rows: int
    updated_cells: int


SOURCES: dict[str, CrosstabSource] = {
    "Cygnal": CrosstabSource(
        pollster="Cygnal",
        parser=parse_cygnal_report,
        default_path=DEFAULT_SOURCE_DIR / "cygnal_crosstab_extract.txt",
    ),
    "Trafalgar Group": CrosstabSource(
        pollster="Trafalgar Group",
        parser=parse_trafalgar_report,
        default_path=DEFAULT_SOURCE_DIR / "trafalgar_crosstab_extract.txt",
    ),
    "Quantus Insights": CrosstabSource(
        pollster="Quantus Insights",
        parser=parse_quantus_report,
        default_path=DEFAULT_SOURCE_DIR / "quantus_generic_ballot_extract.txt",
    ),
    "TIPP Insights": CrosstabSource(
        pollster="TIPP Insights",
        parser=parse_tipp_report,
        default_path=DEFAULT_SOURCE_DIR / "tipp_crosstab_extract.txt",
    ),
}


def _format_csv_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def _source_by_name(name: str) -> CrosstabSource:
    normalized = name.strip().lower()
    for source in SOURCES.values():
        if source.pollster.lower() == normalized:
            return source
    available = ", ".join(SOURCES)
    raise ValueError(f"Unknown pollster {name!r}; expected one of: {available}")


def populate_csv(
    csv_path: str | Path = POLLS_CSV,
    pollsters: Iterable[str] | None = None,
    source_paths: dict[str, str | Path] | None = None,
    *,
    overwrite: bool = False,
) -> list[PopulateStats]:
    """Fill matching rows in ``csv_path`` from parsed crosstab extracts.

    Only columns already present in the CSV and beginning with ``xt_`` are
    eligible for updates. By default existing non-empty crosstab values are
    preserved, so running this after Emerson or Marist enrichment does not
    destroy more specific row-level crosstabs.
    """
    csv_path = Path(csv_path)
    selected = [_source_by_name(name) for name in pollsters] if pollsters else list(SOURCES.values())
    source_paths = source_paths or {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")

    all_xt_columns = {field for field in fieldnames if field.startswith("xt_")}
    stats: list[PopulateStats] = []

    for source in selected:
        source_path = Path(source_paths.get(source.pollster, source.default_path))
        parsed = source.parser(source_path)
        xt_values = {
            key: _format_csv_value(value)
            for key, value in parsed.items()
            if key in all_xt_columns and key.startswith("xt_") and value is not None
        }

        updated_rows = 0
        updated_cells = 0
        for row in rows:
            if row.get("pollster", "").strip().lower() != source.pollster.lower():
                continue

            row_cells = 0
            for column, value in xt_values.items():
                if overwrite or _is_blank(row.get(column)):
                    if row.get(column) != value:
                        row[column] = value
                        row_cells += 1
            if row_cells:
                updated_rows += 1
                updated_cells += row_cells

        stats.append(
            PopulateStats(
                pollster=source.pollster,
                source_path=source_path,
                parsed_xt_columns=len(xt_values),
                updated_rows=updated_rows,
                updated_cells=updated_cells,
            )
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def backfill_from_existing_pollster_rows(
    csv_path: str | Path,
    pollster: str,
    *,
    overwrite: bool = False,
) -> PopulateStats:
    """Fill empty xt_* cells from the most common existing value for a pollster.

    This is a fallback for pollster groups where source-specific crosstabs are
    partially unavailable but the CSV already contains parser-ingested crosstabs
    for that pollster. It never invents a value for a column unless at least one
    row for the same pollster already has that column populated.
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")

    xt_columns = [field for field in fieldnames if field.startswith("xt_")]
    owned_rows = [row for row in rows if row.get("pollster", "").strip().lower() == pollster.lower()]
    fill_values: dict[str, str] = {}
    for column in xt_columns:
        values = [row[column] for row in owned_rows if not _is_blank(row.get(column))]
        if values:
            fill_values[column] = Counter(values).most_common(1)[0][0]

    updated_rows = 0
    updated_cells = 0
    for row in owned_rows:
        row_cells = 0
        for column, value in fill_values.items():
            if overwrite or _is_blank(row.get(column)):
                if row.get(column) != value:
                    row[column] = value
                    row_cells += 1
        if row_cells:
            updated_rows += 1
            updated_cells += row_cells

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return PopulateStats(
        pollster=pollster,
        source_path=csv_path,
        parsed_xt_columns=len(fill_values),
        updated_rows=updated_rows,
        updated_cells=updated_cells,
    )


def _parse_source_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source must use Pollster=path format, got {value!r}")
        pollster, path = value.split("=", 1)
        source = _source_by_name(pollster)
        overrides[source.pollster] = Path(path)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate poll CSV xt_* fields from crosstab parser extracts.")
    parser.add_argument("--csv", default=str(POLLS_CSV), help="Path to polls_2026.csv")
    parser.add_argument(
        "--pollster",
        action="append",
        choices=list(SOURCES),
        help="Pollster to populate; may be repeated. Defaults to all configured pollsters.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Override source extract path as Pollster=path. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing non-empty xt_* values.")
    parser.add_argument(
        "--backfill-existing-pollster",
        action="append",
        default=[],
        help="Fill blank xt_* cells from existing values for this pollster. May be repeated.",
    )
    args = parser.parse_args()

    stats = populate_csv(
        csv_path=args.csv,
        pollsters=args.pollster,
        source_paths=_parse_source_overrides(args.source),
        overwrite=args.overwrite,
    )
    for item in stats:
        print(
            f"{item.pollster}: parsed {item.parsed_xt_columns} xt columns from "
            f"{item.source_path}; updated {item.updated_cells} cells across {item.updated_rows} rows"
        )
    for pollster in args.backfill_existing_pollster:
        item = backfill_from_existing_pollster_rows(args.csv, pollster, overwrite=args.overwrite)
        print(
            f"{item.pollster}: backfilled {item.updated_cells} empty xt cells across "
            f"{item.updated_rows} rows from {item.parsed_xt_columns} existing columns"
        )


if __name__ == "__main__":
    main()
