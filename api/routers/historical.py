# api/routers/historical.py
"""Historical presidential election results endpoint.

Serves county-level Democratic two-party vote share for past presidential
elections (2012, 2016, 2020) so the frontend can render comparison overlay
layers alongside the current forecast choropleth.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Presidential years available as parquet files under data/assembled/
_AVAILABLE_YEARS = {2012, 2016, 2020}

_DATA_DIR = Path(os.environ.get("WETHERVANE_DATA_DIR", "data"))
_ASSEMBLED = _DATA_DIR / "assembled"

router = APIRouter(tags=["historical"])


class CountyHistoricalRow(BaseModel):
    """County-level presidential election result for a single year."""

    county_fips: str
    dem_share: float
    # Two-party total vote count (may be None if not present in source data)
    total_votes: int | None


class HistoricalElectionResponse(BaseModel):
    """Response envelope for a single historical presidential election year."""

    year: int
    counties: list[CountyHistoricalRow]


def _load_presidential_year(year: int) -> list[CountyHistoricalRow]:
    """Read parquet and return county dem_share rows for the given year.

    Uses the medsl_county_presidential_{year}.parquet files assembled from
    MEDSL Harvard Dataverse data.  The 2024 data lives in a differently-named
    file; this endpoint only covers 2012/2016/2020 for the overlay feature.

    Rows where dem_share is NaN (missing data) are dropped.
    The synthetic '00000' county_fips (aggregate row sometimes present in MEDSL
    exports) is excluded because it doesn't correspond to a real county polygon.
    """
    path = _ASSEMBLED / f"medsl_county_presidential_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No presidential data file for year {year}: {path}")

    df = pd.read_parquet(
        path,
        columns=[
            "county_fips",
            f"pres_dem_share_{year}",
            f"pres_total_{year}",
        ],
    )

    # Drop aggregate/synthetic rows and rows with missing dem_share
    df = df[df["county_fips"] != "00000"]
    df = df.dropna(subset=[f"pres_dem_share_{year}"])

    rows: list[CountyHistoricalRow] = []
    for _, row in df.iterrows():
        total_raw = row[f"pres_total_{year}"]
        rows.append(
            CountyHistoricalRow(
                county_fips=str(row["county_fips"]),
                dem_share=float(row[f"pres_dem_share_{year}"]),
                total_votes=int(total_raw) if pd.notna(total_raw) else None,
            )
        )
    return rows


@router.get(
    "/historical/presidential/{year}",
    response_model=HistoricalElectionResponse,
    summary="County-level presidential results for a past election year",
    description=(
        "Returns the Democratic two-party vote share (dem_share) for every county "
        "in the requested presidential election year. "
        "Available years: 2012, 2016, 2020."
    ),
)
def get_historical_presidential(year: int) -> HistoricalElectionResponse:
    """Return county-level Dem share for a past presidential election.

    Used by the frontend to render a semi-transparent historical comparison
    overlay on the stained-glass map.
    """
    if year not in _AVAILABLE_YEARS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Year {year} is not available. "
                f"Supported years: {sorted(_AVAILABLE_YEARS)}"
            ),
        )
    try:
        counties = _load_presidential_year(year)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return HistoricalElectionResponse(year=year, counties=counties)
