"""CSV population tests for pollster crosstab parser output."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from populate_pollster_crosstabs import backfill_from_existing_pollster_rows, populate_csv  # noqa: E402


FIELDNAMES = [
    "race",
    "geography",
    "geo_level",
    "dem_share",
    "n_sample",
    "date",
    "pollster",
    "notes",
    "methodology",
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
    "xt_vote_race_white",
    "xt_vote_race_black",
    "xt_vote_race_hispanic",
    "xt_vote_race_asian",
    "xt_vote_education_college",
    "xt_vote_education_noncollege",
    "xt_vote_age_senior",
    "xt_vote_gender_men",
    "xt_vote_gender_women",
    "xt_vote_party_democrat",
    "xt_vote_party_independent",
    "xt_vote_party_republican",
]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize(
    ("pollster", "expected_columns"),
    [
        ("Cygnal", ["xt_race_white", "xt_education_college", "xt_vote_gender_men"]),
        ("Trafalgar Group", ["xt_race_white", "xt_education_college", "xt_vote_age_senior"]),
        ("Quantus Insights", ["xt_race_white", "xt_urbanicity_urban", "xt_vote_age_senior"]),
        ("TIPP Insights", ["xt_race_white", "xt_vote_gender_women", "xt_vote_party_independent"]),
    ],
)
def test_populate_csv_fills_pollster_xt_columns(tmp_path: Path, pollster: str, expected_columns: list[str]) -> None:
    csv_path = tmp_path / "polls_2026.csv"
    _write_csv(
        csv_path,
        [
            {
                "race": "2026 Generic Ballot",
                "geography": "US",
                "geo_level": "national",
                "dem_share": "0.52",
                "n_sample": "999",
                "date": "2026-04-24",
                "pollster": pollster,
                "notes": "fresh topline",
                "methodology": "original method",
            },
            {
                "race": "2026 Generic Ballot",
                "geography": "US",
                "geo_level": "national",
                "dem_share": "0.50",
                "n_sample": "700",
                "date": "2026-04-24",
                "pollster": "Other Pollster",
                "notes": "control",
            },
        ],
    )

    stats = populate_csv(csv_path=csv_path, pollsters=[pollster])
    rows = _read_rows(csv_path)
    target = rows[0]
    control = rows[1]

    assert stats[0].updated_rows == 1
    assert stats[0].updated_cells >= len(expected_columns)
    for column in expected_columns:
        assert target[column] != ""

    assert target["n_sample"] == "999"
    assert target["date"] == "2026-04-24"
    assert target["methodology"] == "original method"
    assert all(control[column] == "" for column in expected_columns)


def test_populate_csv_preserves_existing_xt_by_default(tmp_path: Path) -> None:
    csv_path = tmp_path / "polls_2026.csv"
    _write_csv(
        csv_path,
        [
            {
                "race": "2026 Generic Ballot",
                "geography": "US",
                "geo_level": "national",
                "dem_share": "0.52",
                "date": "2026-04-24",
                "pollster": "Cygnal",
                "xt_race_white": "0.123456",
            },
        ],
    )

    populate_csv(csv_path=csv_path, pollsters=["Cygnal"])
    row = _read_rows(csv_path)[0]

    assert row["xt_race_white"] == "0.123456"
    assert row["xt_vote_race_white"] != ""


def test_backfill_from_existing_pollster_rows_uses_same_pollster_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "polls_2026.csv"
    _write_csv(
        csv_path,
        [
            {
                "race": "2026 AZ Governor",
                "geography": "AZ",
                "geo_level": "state",
                "dem_share": "0.52",
                "date": "2026-04-24",
                "pollster": "Emerson College",
                "xt_race_white": "0.712000",
                "xt_vote_race_white": "0.416000",
            },
            {
                "race": "2026 Generic Ballot",
                "geography": "US",
                "geo_level": "national",
                "dem_share": "0.50",
                "date": "2026-04-26",
                "pollster": "Emerson College",
            },
            {
                "race": "2026 Generic Ballot",
                "geography": "US",
                "geo_level": "national",
                "dem_share": "0.50",
                "date": "2026-04-26",
                "pollster": "Other Pollster",
            },
        ],
    )

    stats = backfill_from_existing_pollster_rows(csv_path, "Emerson College")
    rows = _read_rows(csv_path)

    assert stats.updated_rows == 1
    assert rows[1]["xt_race_white"] == "0.712000"
    assert rows[1]["xt_vote_race_white"] == "0.416000"
    assert rows[2]["xt_race_white"] == ""
