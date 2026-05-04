"""Build multi-year county shift vectors.

Training presidential pairs (18 × 3 dims = 54 dims):
  Pre-2000 Algara pairs: 1948→1952, 1952→1956, ..., 1996→2000
  Post-2000 MEDSL pairs: 2000→2004, 2004→2008, 2008→2012, 2012→2016, 2016→2020

Governor training dims (7 pairs × 3 dims = 21 dims):
  1994→1998, 1998→2002, 2002→2006, 2006→2010, 2010→2014, 2014→2018, 2018→2022

Senate training dims (up to 9 pairs × 3 dims = 27 dims, when data available):
  Pairs align same Senate seat across 6-year cycles (same class):
    2002→2008, 2004→2010, 2006→2012, 2008→2014, 2010→2016,
    2012→2018, 2014→2020, 2016→2022, 2018→2024

Holdout (3 dims — never used during clustering):
  Presidential 2020→2024

Note: All AL governor cycles are fully contested. No structural zeros needed
(AL 2018 governor: Kay Ivey vs Walter Maddox, Ivey won ~60/40 — real data).

Senate counties that are uncontested or missing are zero-filled (logged).

Output:
  data/shifts/county_shifts_multiyear.parquet
  Columns: county_fips + pres/gov/senate training dims + 3 holdout shift dims
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_DIR = PROJECT_ROOT / "data" / "shifts"

AL_FIPS_PREFIX = _cfg.AL_FIPS_PREFIX
EPSILON = _cfg.LOGODDS_EPSILON  # clip dem_share to [EPSILON, 1-EPSILON] before logit

# ── Column name constants ─────────────────────────────────────────────────────

PRES_PAIRS = _cfg.PRES_PAIRS
GOV_PAIRS = _cfg.GOV_PAIRS
SENATE_PAIRS = _cfg.SENATE_PAIRS
HOLDOUT_PAIRS = _cfg.HOLDOUT_PRES_PAIRS

TRAINING_SHIFT_COLS: list[str] = []
for a, b in PRES_PAIRS:
    TRAINING_SHIFT_COLS += [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]
for a, b in GOV_PAIRS:
    TRAINING_SHIFT_COLS += [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}", f"gov_turnout_shift_{a}_{b}"]
for a, b in SENATE_PAIRS:
    TRAINING_SHIFT_COLS += [f"sen_d_shift_{a}_{b}", f"sen_r_shift_{a}_{b}", f"sen_turnout_shift_{a}_{b}"]

HOLDOUT_SHIFT_COLS: list[str] = []
for a, b in HOLDOUT_PAIRS:
    HOLDOUT_SHIFT_COLS += [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]


# ── Core computation ──────────────────────────────────────────────────────────

def _dem_share_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "dem_share" in c)

def _total_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "_total_" in c)


def _logodds_shift(later_share: pd.Series, earlier_share: pd.Series) -> pd.Series:
    """Log-odds shift: logit(later) - logit(earlier), with epsilon clipping.

    Clips dem_share to [EPSILON, 1-EPSILON] before applying logit to avoid
    infinite values for counties at or near 0% / 100% Democrat.
    Turnout shifts remain as raw proportional change (not log-odds) since
    turnout is not bounded [0, 1] the same way vote share is.
    """
    later_clipped = later_share.clip(EPSILON, 1 - EPSILON)
    earlier_clipped = earlier_share.clip(EPSILON, 1 - EPSILON)
    return np.log(later_clipped / (1 - later_clipped)) - np.log(earlier_clipped / (1 - earlier_clipped))


def compute_pres_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two presidential election DataFrames.

    D and R shifts are log-odds (logit scale). Turnout shift is raw proportional.
    """
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = _logodds_shift(merged[late_dem], merged[early_dem])
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    return pd.DataFrame({
        "county_fips": merged["county_fips"].values,
        f"pres_d_shift_{a}_{b}": d_shift.values,
        f"pres_r_shift_{a}_{b}": r_shift.values,
        f"pres_turnout_shift_{a}_{b}": t_shift.values,
    })


def compute_gov_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two governor election DataFrames.

    D and R shifts are log-odds (logit scale). Turnout shift is raw proportional.
    """
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = _logodds_shift(merged[late_dem], merged[early_dem])
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    return pd.DataFrame({
        "county_fips": merged["county_fips"].values,
        f"gov_d_shift_{a}_{b}": d_shift.values,
        f"gov_r_shift_{a}_{b}": r_shift.values,
        f"gov_turnout_shift_{a}_{b}": t_shift.values,
    })


def compute_senate_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two Senate election DataFrames.

    Senate shifts pair the same seat class across a 6-year cycle
    (e.g. 2002→2008 is the same Class II seat in each state).

    D and R shifts are log-odds (logit scale). Turnout shift is raw proportional.
    Only counties present in both years survive (inner join); counties missing
    from either cycle are zero-filled by build_multiyear_shifts.
    """
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = _logodds_shift(merged[late_dem], merged[early_dem])
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    return pd.DataFrame({
        "county_fips": merged["county_fips"].values,
        f"sen_d_shift_{a}_{b}": d_shift.values,
        f"sen_r_shift_{a}_{b}": r_shift.values,
        f"sen_turnout_shift_{a}_{b}": t_shift.values,
    })


def build_multiyear_shifts(
    spine: pd.DataFrame,
    pres_pairs: list,
    gov_pairs: list,
    senate_pairs: list | None = None,
) -> pd.DataFrame:
    """Join all shift pairs onto the county spine.

    pres_pairs:    list of (a_str, b_str, early_df, late_df)
    gov_pairs:     list of (a_str, b_str, early_df, late_df)
    senate_pairs:  list of (a_str, b_str, early_df, late_df), or None

    Missing pairs produce zero-filled columns (logged as warning).
    """
    if senate_pairs is None:
        senate_pairs = []

    result = spine[["county_fips"]].copy()

    all_cols = TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS
    for col in all_cols:
        result[col] = 0.0

    for a, b, early, late in pres_pairs:
        shifts = compute_pres_shift(early, late, a, b)
        cols = [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]
        merged = result[["county_fips"]].merge(shifts[["county_fips"] + cols], on="county_fips", how="left")
        for col in cols:
            result[col] = merged[col].fillna(0.0).values

    for a, b, early, late in gov_pairs:
        shifts = compute_gov_shift(early, late, a, b)
        cols = [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}", f"gov_turnout_shift_{a}_{b}"]
        merged = result[["county_fips"]].merge(shifts[["county_fips"] + cols], on="county_fips", how="left")
        for col in cols:
            result[col] = merged[col].fillna(0.0).values

    for a, b, early, late in senate_pairs:
        shifts = compute_senate_shift(early, late, a, b)
        cols = [f"sen_d_shift_{a}_{b}", f"sen_r_shift_{a}_{b}", f"sen_turnout_shift_{a}_{b}"]
        merged = result[["county_fips"]].merge(shifts[["county_fips"] + cols], on="county_fips", how="left")
        for col in cols:
            result[col] = merged[col].fillna(0.0).values

    return result[["county_fips"] + TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS]


def _load(filename: str) -> pd.DataFrame | None:
    path = ASSEMBLED_DIR / filename
    if not path.exists():
        log.warning("Missing: %s — will zero-fill this pair", path.name)
        return None
    return pd.read_parquet(path)


def main() -> None:
    """Load all election parquets and build multi-year shifts."""
    # Spine: use medsl_county_presidential_2024 (new format file)
    spine_df = _load("medsl_county_presidential_2024.parquet")
    if spine_df is None:
        # Fallback to old-format file
        spine_df = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")
    spine = spine_df[["county_fips"]].copy()
    log.info("County spine: %d counties", len(spine))

    # ── Presidential pairs ────────────────────────────────────────────────────
    pres_file = dict(_cfg.PRES_FILES)
    pres_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in pres_file.items()}
    # Fallback for 24 if new file not present
    if pres_dfs.get("24") is None:
        pres_dfs["24"] = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")

    pres_pairs = []
    for a, b in PRES_PAIRS + HOLDOUT_PAIRS:
        early, late = pres_dfs.get(a), pres_dfs.get(b)
        if early is not None and late is not None:
            pres_pairs.append((a, b, early, late))
        else:
            log.warning("Skipping pres pair %s→%s (data missing)", a, b)

    # ── Governor pairs ────────────────────────────────────────────────────────
    gov_file = dict(_cfg.GOV_FILES)
    gov_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in gov_file.items()}

    gov_pairs = []
    for a, b in GOV_PAIRS:
        early, late = gov_dfs.get(a), gov_dfs.get(b)
        if early is not None and late is not None:
            gov_pairs.append((a, b, early, late))
        else:
            log.warning("Skipping gov pair %s→%s (data missing)", a, b)

    # ── Senate pairs ──────────────────────────────────────────────────────────
    senate_file = dict(_cfg.SENATE_FILES)
    senate_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in senate_file.items()}

    senate_pairs_loaded = []
    for a, b in SENATE_PAIRS:
        early, late = senate_dfs.get(a), senate_dfs.get(b)
        if early is not None and late is not None:
            senate_pairs_loaded.append((a, b, early, late))
        else:
            log.warning("Skipping senate pair %s→%s (data missing)", a, b)

    # ── Build ─────────────────────────────────────────────────────────────────
    log.info(
        "Building multi-year shifts: %d pres pairs, %d gov pairs, %d senate pairs",
        len(pres_pairs), len(gov_pairs), len(senate_pairs_loaded),
    )
    shifts = build_multiyear_shifts(spine, pres_pairs, gov_pairs, senate_pairs_loaded)

    SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SHIFTS_DIR / "county_shifts_multiyear.parquet"
    shifts.to_parquet(out, index=False)
    n_senate_dims = len(senate_pairs_loaded) * 3
    log.info(
        "Saved → %s | %d counties | %d training dims (%d senate) + %d holdout dims",
        out, len(shifts), len(TRAINING_SHIFT_COLS), n_senate_dims, len(HOLDOUT_SHIFT_COLS),
    )

    zero_cols = [c for c in TRAINING_SHIFT_COLS if shifts[c].abs().max() < 1e-10]
    if zero_cols:
        raise ValueError(
            f"All-zero training columns detected — data is missing for these shift pairs: "
            f"{zero_cols}. Re-run the relevant fetch scripts before building shifts."
        )


if __name__ == "__main__":
    main()
