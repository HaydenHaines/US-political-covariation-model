"""Build tract-level feature matrix from interpolated votes, ACS, and RCMS data.

Computes electoral shifts (log-odds), demographic ratios, and religion proxies
for all census tracts. Outputs a single parquet with GEOID index.

CLI: python -m src.tracts.build_tract_features
Output: data/tracts/tract_features.parquet
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPSILON = _cfg.LOGODDS_EPSILON  # 0.01


# ── Helpers ──────────────────────────────────────────────────────────────────


def _logodds(p: float | pd.Series) -> float | pd.Series:
    """Compute log-odds (logit) with epsilon clipping."""
    if isinstance(p, pd.Series):
        p_clip = p.clip(EPSILON, 1 - EPSILON)
        return np.log(p_clip / (1 - p_clip))
    p_clip = max(EPSILON, min(p, 1 - EPSILON))
    return math.log(p_clip / (1 - p_clip))


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numerator by denominator; return NaN where denominator is zero or NaN."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _state_from_geoid(geoid: str, state_fips_map: dict[str, str]) -> str | None:
    """Extract state abbreviation from tract GEOID (first 2 chars = state FIPS)."""
    prefix = geoid[:2]
    return state_fips_map.get(prefix)


# ── Electoral features ───────────────────────────────────────────────────────


def _build_presidential_levels(
    tract_votes: dict[str, pd.DataFrame],
    tract_areas: pd.Series | None,
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Merge presidential dem-share, turnout, and vote-density columns into result."""
    for key, df in tract_votes.items():
        parts = key.split("_")
        race = parts[0]
        year = int(parts[1])

        if race != "president":
            continue

        # Dem share level
        col_lean = f"pres_dem_share_{year}"
        merge_df = df[["GEOID", "dem_share"]].rename(columns={"dem_share": col_lean})
        result = result.merge(merge_df, on="GEOID", how="left")

        # Turnout level
        col_turnout = f"turnout_{year}"
        merge_df = df[["GEOID", "votes_total"]].rename(columns={"votes_total": col_turnout})
        result = result.merge(merge_df, on="GEOID", how="left")

        # Vote density (votes per sq km)
        if tract_areas is not None:
            col_density = f"vote_density_{year}"
            density_df = df[["GEOID", "votes_total"]].copy()
            density_df = density_df.set_index("GEOID")
            density_df[col_density] = density_df["votes_total"] / tract_areas
            density_df = density_df[[col_density]].reset_index()
            result = result.merge(density_df, on="GEOID", how="left")

    return result


def _build_presidential_shifts(
    tract_votes: dict[str, pd.DataFrame],
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Merge presidential log-odds shifts and turnout shifts into result.

    Presidential shifts are NOT state-centered — they carry cross-state
    signal and must remain raw for the clustering model.
    """
    pres_pairs = [("2008", "2012"), ("2012", "2016"), ("2016", "2020"), ("2020", "2024")]
    for early_yr, late_yr in pres_pairs:
        early_key = f"president_{early_yr}"
        late_key = f"president_{late_yr}"
        if early_key not in tract_votes or late_key not in tract_votes:
            continue

        early_df = tract_votes[early_key]
        late_df = tract_votes[late_key]

        merged = early_df[["GEOID", "dem_share", "votes_total"]].merge(
            late_df[["GEOID", "dem_share", "votes_total"]],
            on="GEOID",
            how="inner",
            suffixes=("_early", "_late"),
        )

        # Dem log-odds shift (positive = Dem gain)
        d_shift = _logodds(merged["dem_share_late"]) - _logodds(merged["dem_share_early"])
        merged[f"pres_shift_{early_yr}_{late_yr}"] = d_shift

        # Proportional turnout shift
        early_total = merged["votes_total_early"].replace(0, np.nan)
        merged[f"pres_turnout_shift_{early_yr}_{late_yr}"] = (
            (merged["votes_total_late"] - merged["votes_total_early"]) / early_total
        )

        shift_cols = [
            f"pres_shift_{early_yr}_{late_yr}",
            f"pres_turnout_shift_{early_yr}_{late_yr}",
        ]
        result = result.merge(merged[["GEOID"] + shift_cols], on="GEOID", how="left")

    # Standalone turnout-shift aliases (mirror the pres_turnout_shift_ columns)
    for early_yr, late_yr in pres_pairs:
        pres_col = f"pres_turnout_shift_{early_yr}_{late_yr}"
        if pres_col in result.columns:
            result[f"turnout_shift_{early_yr}_{late_yr}"] = result[pres_col]

    return result


def _state_center_series(raw_shift: pd.Series, state_fips: pd.Series) -> pd.Series:
    """Subtract each state's mean from raw_shift, in-place by state group.

    Tracts with only one observation per state are left uncentered (stable
    mean is not computable with N=1).
    """
    centered = raw_shift.copy()
    for st in state_fips.unique():
        mask = state_fips == st
        st_vals = raw_shift[mask]
        valid = st_vals.dropna()
        if len(valid) > 1:
            centered[mask] = st_vals - valid.mean()
    return centered


def _build_offcycle_shifts(
    tract_votes: dict[str, pd.DataFrame],
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Merge state-centered off-cycle (governor, senate, house) shifts into result.

    Off-cycle shifts are state-centered by subtracting the within-state mean.
    This is a proxy for candidate-effect removal: different candidates in each
    state add state-level bias that must be removed before cross-state clustering.
    """
    _OFFCYCLE_RACE_PREFIX = {
        "governor": "gov",
        "senate": "sen",
        "house": "house",
    }

    # Build (race, early_year, late_year) tuples for all consecutive pairs
    offcycle_pairs: list[tuple[str, str, str]] = []
    for race in _OFFCYCLE_RACE_PREFIX:
        race_years = sorted(
            int(k.split("_")[1])
            for k in tract_votes
            if k.startswith(f"{race}_")
        )
        for i in range(len(race_years) - 1):
            offcycle_pairs.append((race, str(race_years[i]), str(race_years[i + 1])))

    for race, early_yr, late_yr in offcycle_pairs:
        early_key = f"{race}_{early_yr}"
        late_key = f"{race}_{late_yr}"
        if early_key not in tract_votes or late_key not in tract_votes:
            continue

        early_df = tract_votes[early_key]
        late_df = tract_votes[late_key]

        merged = early_df[["GEOID", "dem_share"]].merge(
            late_df[["GEOID", "dem_share"]],
            on="GEOID",
            how="inner",
            suffixes=("_early", "_late"),
        )

        prefix = _OFFCYCLE_RACE_PREFIX[race]
        col_name = f"{prefix}_shift_{early_yr}_{late_yr}"

        raw_shift = _logodds(merged["dem_share_late"]) - _logodds(merged["dem_share_early"])
        state_fips = merged["GEOID"].str[:2]
        merged[col_name] = _state_center_series(raw_shift, state_fips)

        result = result.merge(merged[["GEOID", col_name]], on="GEOID", how="left")

    return result


def _build_split_ticket(
    tract_votes: dict[str, pd.DataFrame],
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Merge abs(pres_dem_share - house_dem_share) split-ticket columns into result."""
    for year in (2016, 2020):
        pres_key = f"president_{year}"
        house_key = f"house_{year}"
        if pres_key not in tract_votes or house_key not in tract_votes:
            continue

        pres_df = tract_votes[pres_key][["GEOID", "dem_share"]].rename(
            columns={"dem_share": "pres_dem"}
        )
        house_df = tract_votes[house_key][["GEOID", "dem_share"]].rename(
            columns={"dem_share": "house_dem"}
        )
        merged = pres_df.merge(house_df, on="GEOID", how="inner")
        merged[f"split_ticket_{year}"] = (merged["pres_dem"] - merged["house_dem"]).abs()
        result = result.merge(
            merged[["GEOID", f"split_ticket_{year}"]], on="GEOID", how="left"
        )

    return result


def build_electoral_features(
    tract_votes: dict[str, pd.DataFrame],
    tract_areas: pd.Series,
    state_fips_map: dict[str, str],
) -> pd.DataFrame:
    """Compute shifts, lean, turnout, density, split-ticket from tract votes.

    Parameters
    ----------
    tract_votes : dict[str, pd.DataFrame]
        Keys like "president_2020", "president_2024", "house_2020", "governor_2018".
        Each DataFrame has columns: GEOID, votes_dem, votes_rep, votes_total, dem_share, state, year, race.
    tract_areas : pd.Series
        GEOID → area in sq km.
    state_fips_map : dict[str, str]
        FIPS prefix → state abbreviation (e.g. "12" → "FL").

    Returns
    -------
    DataFrame with GEOID + all electoral feature columns.
    """
    # Collect all GEOIDs across all vote DataFrames
    all_geoids = sorted(
        set(geoid for df in tract_votes.values() for geoid in df["GEOID"].values)
    )
    result = pd.DataFrame({"GEOID": all_geoids})

    result = _build_presidential_levels(tract_votes, tract_areas, result)
    result = _build_presidential_shifts(tract_votes, result)
    result = _build_offcycle_shifts(tract_votes, result)
    result = _build_split_ticket(tract_votes, result)

    return result


# ── Demographic features ────────────────────────────────────────────────────


def build_demographic_features(tract_acs: pd.DataFrame) -> pd.DataFrame:
    """Compute all demographic features from ACS tract data.

    Parameters
    ----------
    tract_acs : pd.DataFrame
        Raw ACS tract data with columns like pop_total, pop_white_nh, etc.

    Returns
    -------
    DataFrame with GEOID + all demographic feature columns.
    """
    df = tract_acs[["GEOID"]].copy()

    pop = tract_acs["pop_total"]

    # Race / ethnicity
    df["pct_white_nh"] = _safe_ratio(tract_acs["pop_white_nh"], pop)
    df["pct_black"] = _safe_ratio(tract_acs["pop_black"], pop)
    df["pct_hispanic"] = _safe_ratio(tract_acs["pop_hispanic"], pop)
    df["pct_asian"] = _safe_ratio(tract_acs["pop_asian"], pop)

    # Foreign born
    df["pct_foreign_born"] = _safe_ratio(tract_acs["pop_foreign_born"], pop)

    # Education
    educ_total = tract_acs["educ_total"]
    ba_plus = tract_acs["educ_bachelors"] + tract_acs["educ_graduate"]
    df["pct_ba_plus"] = _safe_ratio(ba_plus, educ_total)
    df["pct_graduate"] = _safe_ratio(tract_acs["educ_graduate"], educ_total)
    df["pct_no_hs"] = _safe_ratio(tract_acs["educ_no_hs"], educ_total)

    # White working class interaction
    df["pct_wwc"] = df["pct_white_nh"] * (1 - df["pct_ba_plus"])

    # Income
    df["median_hh_income"] = tract_acs["median_hh_income"]
    df["poverty_rate"] = tract_acs["poverty_rate"]
    df["gini"] = tract_acs["gini"]

    # Housing
    housing_units = tract_acs["housing_units"]
    df["pct_owner_occupied"] = _safe_ratio(tract_acs["housing_owner"], housing_units)
    df["median_home_value"] = tract_acs["median_home_value"]
    df["pct_multi_unit"] = _safe_ratio(tract_acs["housing_multi_unit"], housing_units)
    df["pct_pre_1960"] = _safe_ratio(tract_acs["housing_pre_1960"], housing_units)

    # Rent burden: annual rent / median income
    df["rent_burden"] = _safe_ratio(tract_acs["median_rent"] * 12, tract_acs["median_hh_income"])

    # Age and household
    df["median_age"] = tract_acs["median_age"]
    df["pct_under_18"] = _safe_ratio(tract_acs["pop_under_18"], pop)
    df["pct_over_65"] = _safe_ratio(tract_acs["pop_over_65"], pop)
    df["pct_single_hh"] = _safe_ratio(tract_acs["hh_single"], tract_acs["hh_total"])

    # Commute
    df["pct_wfh"] = _safe_ratio(tract_acs["commute_wfh"], pop)
    df["mean_commute_time"] = tract_acs["mean_commute_time"]
    df["pct_no_vehicle"] = _safe_ratio(tract_acs["pop_no_vehicle"], pop)

    # Military
    df["pct_veteran"] = _safe_ratio(tract_acs["pop_veteran"], pop)

    return df.reset_index(drop=True)


# ── Religion features ────────────────────────────────────────────────────────


def build_religion_features(
    rcms_county: pd.DataFrame,
    tract_geoids: pd.Series,
) -> pd.DataFrame:
    """Map county RCMS to tracts via FIPS prefix (county proxy).

    Parameters
    ----------
    rcms_county : pd.DataFrame
        County-level RCMS data with county_fips and religion columns.
    tract_geoids : pd.Series
        Series of tract GEOID strings.

    Returns
    -------
    DataFrame with GEOID + religion feature columns.
    """
    religion_cols = [
        c for c in ["evangelical_share", "catholic_share", "black_protestant_share",
                    "adherence_rate", "religious_adherence_rate"]
        if c in rcms_county.columns
    ]

    df = pd.DataFrame({"GEOID": tract_geoids.values})
    # Extract county FIPS (first 5 chars of tract GEOID)
    df["county_fips"] = df["GEOID"].str[:5]

    df = df.merge(
        rcms_county[["county_fips"] + religion_cols],
        on="county_fips",
        how="left",
    )

    return df[["GEOID"] + religion_cols]


# ── Full pipeline ────────────────────────────────────────────────────────────


def _load_tract_votes(tracts_dir: Path) -> dict[str, pd.DataFrame]:
    """Load tract vote DataFrames from DRA stacked file and/or per-election parquets.

    Supports two formats:
      1. DRA stacked file: tract_votes_dra.parquet — single file with `race` and `year`
         columns and `tract_geoid` as the key.
      2. Per-election files: tract_votes_{state}_{year}_{race}.parquet — one file per
         state/year/race combination with a `GEOID` key column.

    Returns a dict keyed by "{race}_{year}".
    """
    tract_votes: dict[str, pd.DataFrame] = {}

    dra_path = tracts_dir / "tract_votes_dra.parquet"
    if dra_path.exists():
        log.info("Loading DRA stacked tract vote file from %s", dra_path)
        dra_df = pd.read_parquet(dra_path)
        # Normalize key column name to GEOID for downstream compatibility
        if "tract_geoid" in dra_df.columns and "GEOID" not in dra_df.columns:
            dra_df = dra_df.rename(columns={"tract_geoid": "GEOID"})
        for (race, year), grp in dra_df.groupby(["race", "year"]):
            tract_votes[f"{race}_{year}"] = grp.reset_index(drop=True)
        log.info("  Loaded %d election datasets from DRA file", len(tract_votes))

    # Also load any per-election parquet files (legacy / supplementary)
    for fpath in sorted(tracts_dir.glob("tract_votes_*.parquet")):
        if fpath.name == "tract_votes_dra.parquet":
            continue  # already handled above
        df = pd.read_parquet(fpath)
        # Filename pattern: tract_votes_{state}_{year}_{race}.parquet
        parts = fpath.stem.split("_")  # ["tract", "votes", state, year, race]
        if len(parts) < 5:
            log.warning("Skipping unrecognized tract vote file: %s", fpath.name)
            continue
        key = f"{parts[4]}_{parts[3]}"  # "{race}_{year}"
        if key in tract_votes:
            tract_votes[key] = pd.concat([tract_votes[key], df], ignore_index=True)
        else:
            tract_votes[key] = df

    return tract_votes


def _load_demographic_features(assembled_dir: Path) -> pd.DataFrame | None:
    """Load and normalize ACS tract demographics; return None if file not found."""
    acs_path = assembled_dir / "acs_tracts_2022.parquet"
    if not acs_path.exists():
        log.warning(
            "ACS tract data not found at %s. "
            "Run 'python -m src.assembly.fetch_acs --tracts' first. "
            "Skipping demographic features.",
            acs_path,
        )
        return None

    log.info("Loading ACS tract data from %s", acs_path)
    acs_df = pd.read_parquet(acs_path)
    # Normalize key column: ACS uses tract_geoid; feature builder expects GEOID
    if "tract_geoid" in acs_df.columns and "GEOID" not in acs_df.columns:
        acs_df = acs_df.rename(columns={"tract_geoid": "GEOID"})
    return build_demographic_features(acs_df)


def _load_religion_features(
    assembled_dir: Path, all_geoids_sorted: list[str]
) -> pd.DataFrame | None:
    """Load RCMS county religion data and map to tracts; return None if not found."""
    rcms_path = assembled_dir / "county_rcms_features.parquet"
    if not rcms_path.exists():
        log.warning("RCMS data not found at %s. Skipping religion features.", rcms_path)
        return None

    log.info("Loading RCMS county data from %s", rcms_path)
    rcms_df = pd.read_parquet(rcms_path)
    return build_religion_features(rcms_df, pd.Series(all_geoids_sorted))


def build_all_features() -> pd.DataFrame:
    """Load all sources, build complete feature matrix, save to data/tracts/tract_features.parquet.

    Returns the combined DataFrame.
    """
    tracts_dir = PROJECT_ROOT / "data" / "tracts"
    assembled_dir = PROJECT_ROOT / "data" / "assembled"
    out_path = tracts_dir / "tract_features.parquet"

    state_fips_map = {v: k for k, v in _cfg.STATES.items()}

    tract_votes = _load_tract_votes(tracts_dir)
    if not tract_votes:
        log.warning("No tract vote files found in %s — skipping electoral features", tracts_dir)
        return pd.DataFrame()

    # Build stub tract_areas (placeholder: TIGER areas not yet integrated)
    all_geoids_sorted = sorted(
        set(geoid for df in tract_votes.values() for geoid in df["GEOID"].values)
    )
    tract_areas = pd.Series(1.0, index=pd.Index(all_geoids_sorted, name="GEOID"))

    log.info("Building electoral features from %d vote datasets", len(tract_votes))
    result = build_electoral_features(tract_votes, tract_areas, state_fips_map)

    demographic = _load_demographic_features(assembled_dir)
    if demographic is not None:
        result = result.merge(demographic, on="GEOID", how="left")

    religion = _load_religion_features(assembled_dir, all_geoids_sorted)
    if religion is not None:
        result = result.merge(religion, on="GEOID", how="left")

    tracts_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    log.info(
        "Saved tract features → %s | %d tracts × %d columns",
        out_path,
        len(result),
        len(result.columns),
    )

    return result


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    build_all_features()


if __name__ == "__main__":
    main()
