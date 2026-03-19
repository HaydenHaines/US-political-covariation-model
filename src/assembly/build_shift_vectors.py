"""
Stage: Build electoral shift vectors per census tract.

Computes 9-dimensional shift vectors for each tract from election returns:
  - pres_d_shift_16_20, pres_r_shift_16_20, pres_turnout_shift_16_20
  - pres_d_shift_20_24, pres_r_shift_20_24, pres_turnout_shift_20_24
  - mid_d_shift_18_22,  mid_r_shift_18_22,  mid_turnout_shift_18_22

D share shift  = later_dem_share - earlier_dem_share
R share shift  = (1 - later_dem_share) - (1 - earlier_dem_share)  [= -D shift in two-party]
Turnout shift  = (later_total - earlier_total) / earlier_total     [proxy; no VAP denominator]

Special cases:
  - 2024 / 2022 election returns are county-level only (MEDSL); every tract in a
    county receives the county-level shift.
  - AL (FIPS "01") had an uncontested 2018 gubernatorial race; all three midterm
    shift dimensions are set to 0.0 (structural zero, not missing).

Inputs  (data/assembled/):
  vest_tracts_2016.parquet  — tract_geoid, pres_dem_share_2016, pres_total_2016
  vest_tracts_2018.parquet  — tract_geoid, gov_dem_share_2018,  gov_total_2018
  vest_tracts_2020.parquet  — tract_geoid, pres_dem_share_2020, pres_total_2020
  medsl_county_2022_governor.parquet   — county_fips, gov_dem_share_2022, gov_total_2022
  medsl_county_2024_president.parquet  — county_fips, pres_dem_share_2024, pres_total_2024

Output:
  data/shifts/tract_shifts.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_DIR = PROJECT_ROOT / "data" / "shifts"

STATES = {"AL": "01", "FL": "12", "GA": "13"}

AL_FIPS = "01"

SHIFT_COLS: list[str] = [
    "pres_d_shift_16_20",
    "pres_r_shift_16_20",
    "pres_turnout_shift_16_20",
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
    "mid_d_shift_18_22",
    "mid_r_shift_18_22",
    "mid_turnout_shift_18_22",
]


# ── Core computation ──────────────────────────────────────────────────────────


def compute_presidential_shifts(
    early_df: pd.DataFrame,
    late_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two presidential election DataFrames.

    Parameters
    ----------
    early_df:
        Tract-level DataFrame with columns ``tract_geoid``, ``pres_dem_share_{early}``,
        ``pres_total_{early}``.
    late_df:
        Tract-level DataFrame with columns ``tract_geoid``, ``pres_dem_share_{late}``,
        ``pres_total_{late}``.
    label:
        Short label used in output column names, e.g. ``"16_20"`` or ``"20_24"``.
        For ``"20_24"`` the late DataFrame may be county-level and already expanded
        before this function is called.

    Returns
    -------
    pd.DataFrame with columns ``tract_geoid``, ``pres_d_shift_{label}``,
    ``pres_r_shift_{label}``, ``pres_turnout_shift_{label}``.
    """
    # Detect column name patterns dynamically from the DataFrames
    early_dem_col = _find_col(early_df, "dem_share")
    early_total_col = _find_col(early_df, "total")
    late_dem_col = _find_col(late_df, "dem_share")
    late_total_col = _find_col(late_df, "total")

    merged = early_df[["tract_geoid", early_dem_col, early_total_col]].merge(
        late_df[["tract_geoid", late_dem_col, late_total_col]],
        on="tract_geoid",
        how="inner",
    )

    d_shift = merged[late_dem_col] - merged[early_dem_col]
    r_shift = (1.0 - merged[late_dem_col]) - (1.0 - merged[early_dem_col])
    # Relative turnout change as proxy for VAP-normalised shift
    early_total = merged[early_total_col].replace(0, float("nan"))
    turnout_shift = (merged[late_total_col] - merged[early_total_col]) / early_total

    result = pd.DataFrame({
        "tract_geoid": merged["tract_geoid"],
        f"pres_d_shift_{label}": d_shift.values,
        f"pres_r_shift_{label}": r_shift.values,
        f"pres_turnout_shift_{label}": turnout_shift.values,
    })
    log.info(
        "compute_presidential_shifts(%s): %d tracts, d_shift mean=%.4f",
        label,
        len(result),
        result[f"pres_d_shift_{label}"].mean(),
    )
    return result


def compute_midterm_shifts(
    vest_2018: pd.DataFrame,
    medsl_2022: pd.DataFrame,
) -> pd.DataFrame:
    """Compute D/R/turnout midterm shifts (2018 → 2022) per tract.

    AL tracts are assigned 0.0 for all three dimensions (structural zero:
    AL 2018 gubernatorial was uncontested).

    2022 data is county-level; every tract in a county receives the same shift.
    County FIPS is derived from ``tract_geoid[:5]``.

    Parameters
    ----------
    vest_2018:
        Tract-level DataFrame with ``tract_geoid``, ``gov_dem_share_2018``,
        ``gov_total_2018``.
    medsl_2022:
        County-level DataFrame with ``county_fips``, ``gov_dem_share_2022``,
        ``gov_total_2022``.

    Returns
    -------
    pd.DataFrame with columns ``tract_geoid``, ``mid_d_shift_18_22``,
    ``mid_r_shift_18_22``, ``mid_turnout_shift_18_22``.
    """
    # Add county_fips key to 2018 tract data for joining against 2022 county data
    tracts = vest_2018.copy()
    tracts["county_fips"] = tracts["tract_geoid"].str[:5]

    merged = tracts.merge(medsl_2022, on="county_fips", how="left")

    d_shift = merged["gov_dem_share_2022"] - merged["gov_dem_share_2018"]
    r_shift = (1.0 - merged["gov_dem_share_2022"]) - (1.0 - merged["gov_dem_share_2018"])
    early_total = merged["gov_total_2018"].replace(0, float("nan"))
    turnout_shift = (merged["gov_total_2022"] - merged["gov_total_2018"]) / early_total

    result = pd.DataFrame({
        "tract_geoid": merged["tract_geoid"],
        "mid_d_shift_18_22": d_shift.values,
        "mid_r_shift_18_22": r_shift.values,
        "mid_turnout_shift_18_22": turnout_shift.values,
    })

    # Structural zero for AL: uncontested 2018 governor race
    al_mask = result["tract_geoid"].str.startswith(AL_FIPS)
    result.loc[al_mask, ["mid_d_shift_18_22", "mid_r_shift_18_22", "mid_turnout_shift_18_22"]] = 0.0

    log.info(
        "compute_midterm_shifts: %d tracts (%d AL zeroed)",
        len(result),
        al_mask.sum(),
    )
    return result


def build_shift_vectors(
    vest_2016: pd.DataFrame,
    vest_2018: pd.DataFrame,
    vest_2020: pd.DataFrame,
    medsl_2022: pd.DataFrame,
    medsl_2024: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all 9 shift dimensions onto the vest_2020 tract spine.

    The ``vest_2020`` tract list is the authoritative spine. All shift
    DataFrames are left-joined onto it. AL tracts will have NaN midterm shifts
    from the join (they are absent from vest_2018) and are filled with 0.0
    (structural zero).

    2024 data is county-level; tracts are matched via ``tract_geoid[:5]``.

    Parameters
    ----------
    vest_2016, vest_2018, vest_2020:
        Tract-level election DataFrames.
    medsl_2022, medsl_2024:
        County-level election DataFrames.

    Returns
    -------
    pd.DataFrame with ``tract_geoid`` + all 9 columns in ``SHIFT_COLS``.
    """
    spine = vest_2020[["tract_geoid"]].copy()

    # ── Presidential 2016 → 2020 (both tract-level) ───────────────────────────
    pres_16_20 = compute_presidential_shifts(vest_2016, vest_2020, "16_20")
    spine = spine.merge(pres_16_20, on="tract_geoid", how="left")

    # ── Presidential 2020 → 2024 (2024 is county-level; expand to tracts) ────
    vest_2020_with_county = vest_2020.copy()
    vest_2020_with_county["county_fips"] = vest_2020_with_county["tract_geoid"].str[:5]

    # Expand 2024 county shares to tract-level by joining on county_fips
    tract_2024 = vest_2020_with_county[["tract_geoid", "county_fips"]].merge(
        medsl_2024, on="county_fips", how="left"
    ).drop(columns=["county_fips"])

    pres_20_24 = compute_presidential_shifts(vest_2020, tract_2024, "20_24")
    spine = spine.merge(pres_20_24, on="tract_geoid", how="left")

    # ── Midterm 2018 → 2022 ───────────────────────────────────────────────────
    mid_shifts = compute_midterm_shifts(vest_2018, medsl_2022)
    spine = spine.merge(mid_shifts, on="tract_geoid", how="left")

    # Fill NaN midterm shifts for AL tracts (structural zero: uncontested 2018)
    mid_cols = ["mid_d_shift_18_22", "mid_r_shift_18_22", "mid_turnout_shift_18_22"]
    spine[mid_cols] = spine[mid_cols].fillna(0.0)

    log.info(
        "build_shift_vectors: %d tracts, %d shift columns",
        len(spine),
        len(SHIFT_COLS),
    )
    assert list(spine.columns) == ["tract_geoid"] + SHIFT_COLS, (
        f"Unexpected columns: {list(spine.columns)}"
    )
    return spine


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_col(df: pd.DataFrame, substring: str) -> str:
    """Return the first column name containing ``substring`` (case-insensitive)."""
    matches = [c for c in df.columns if substring.lower() in c.lower()]
    if not matches:
        raise ValueError(
            f"No column containing '{substring}' found. Available: {list(df.columns)}"
        )
    return matches[0]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Load parquets, compute shift vectors, write output."""
    log.info("Loading assembled election data...")

    vest_2016 = pd.read_parquet(ASSEMBLED_DIR / "vest_tracts_2016.parquet")
    vest_2018 = pd.read_parquet(ASSEMBLED_DIR / "vest_tracts_2018.parquet")
    vest_2020 = pd.read_parquet(ASSEMBLED_DIR / "vest_tracts_2020.parquet")
    medsl_2022 = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2022_governor.parquet")
    medsl_2024 = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")

    log.info(
        "Loaded: 2016=%d, 2018=%d, 2020=%d tracts; 2022=%d, 2024=%d counties",
        len(vest_2016), len(vest_2018), len(vest_2020), len(medsl_2022), len(medsl_2024),
    )

    shifts = build_shift_vectors(vest_2016, vest_2018, vest_2020, medsl_2022, medsl_2024)

    output_path = SHIFTS_DIR / "tract_shifts.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shifts.to_parquet(output_path, index=False)

    log.info(
        "Saved → %s | %d tracts | %d shift dimensions",
        output_path,
        len(shifts),
        len(SHIFT_COLS),
    )


if __name__ == "__main__":
    main()
