"""Feature registry for tract-level features.

Tags every feature by category (electoral/demographic/religion) and subcategory,
plus source and source_year for config-driven selection and holdout exclusion.

Usage:
    from src.tracts.feature_registry import REGISTRY, select_features
    electoral_names = select_features(category="electoral")
    no_2024 = select_features(exclude_year=2024)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str              # column name in feature matrix
    category: str          # "electoral", "demographic", "religion"
    subcategory: str       # "presidential_shifts", "race_ethnicity", etc.
    source: str            # "vest", "nyt", "acs_tract", "rcms_county"
    source_year: int | None  # year this feature is derived from (for holdout exclusion)
    description: str


# ── Electoral features (~27) ─────────────────────────────────────────────────

_ELECTORAL: list[FeatureSpec] = [
    # Presidential shifts (log-odds)
    FeatureSpec("pres_d_shift_16_20", "electoral", "presidential_shifts", "vest", 2020, "Dem log-odds shift 2016→2020"),
    FeatureSpec("pres_r_shift_16_20", "electoral", "presidential_shifts", "vest", 2020, "Rep log-odds shift 2016→2020"),
    FeatureSpec("pres_turnout_shift_16_20", "electoral", "presidential_shifts", "vest", 2020, "Turnout proportional change 2016→2020"),
    FeatureSpec("pres_d_shift_20_24", "electoral", "presidential_shifts", "nyt", 2024, "Dem log-odds shift 2020→2024"),
    FeatureSpec("pres_r_shift_20_24", "electoral", "presidential_shifts", "nyt", 2024, "Rep log-odds shift 2020→2024"),
    FeatureSpec("pres_turnout_shift_20_24", "electoral", "presidential_shifts", "nyt", 2024, "Turnout proportional change 2020→2024"),

    # Presidential lean (dem_share levels)
    FeatureSpec("pres_dem_share_2016", "electoral", "presidential_lean", "vest", 2016, "Dem vote share 2016 president"),
    FeatureSpec("pres_dem_share_2020", "electoral", "presidential_lean", "vest", 2020, "Dem vote share 2020 president"),
    FeatureSpec("pres_dem_share_2024", "electoral", "presidential_lean", "nyt", 2024, "Dem vote share 2024 president"),

    # Turnout level (raw total votes)
    FeatureSpec("turnout_2016", "electoral", "turnout_level", "vest", 2016, "Total votes cast 2016 president"),
    FeatureSpec("turnout_2020", "electoral", "turnout_level", "vest", 2020, "Total votes cast 2020 president"),
    FeatureSpec("turnout_2024", "electoral", "turnout_level", "nyt", 2024, "Total votes cast 2024 president"),

    # Turnout shift (proportional change in total votes)
    FeatureSpec("turnout_shift_16_20", "electoral", "turnout_shift", "vest", 2020, "Turnout proportional change 2016→2020"),
    FeatureSpec("turnout_shift_20_24", "electoral", "turnout_shift", "nyt", 2024, "Turnout proportional change 2020→2024"),

    # Vote density (votes / sq km)
    FeatureSpec("vote_density_2020", "electoral", "vote_density", "vest", 2020, "Votes per sq km 2020"),
    FeatureSpec("vote_density_2024", "electoral", "vote_density", "nyt", 2024, "Votes per sq km 2024"),

    # House shifts (state-centered)
    FeatureSpec("house_d_shift_16_18_sc", "electoral", "house_shifts", "vest", 2018, "House Dem shift 2016→2018 (state-centered)"),
    FeatureSpec("house_r_shift_16_18_sc", "electoral", "house_shifts", "vest", 2018, "House Rep shift 2016→2018 (state-centered)"),
    FeatureSpec("house_d_shift_18_20_sc", "electoral", "house_shifts", "vest", 2020, "House Dem shift 2018→2020 (state-centered)"),
    FeatureSpec("house_r_shift_18_20_sc", "electoral", "house_shifts", "vest", 2020, "House Rep shift 2018→2020 (state-centered)"),

    # Split ticket
    FeatureSpec("split_ticket_2016", "electoral", "split_ticket", "vest", 2016, "abs(pres_dem - house_dem) 2016"),
    FeatureSpec("split_ticket_2020", "electoral", "split_ticket", "vest", 2020, "abs(pres_dem - house_dem) 2020"),

    # Donor density (county proxy)
    FeatureSpec("donor_density", "electoral", "donor_density", "fec", None, "FEC donor density (county proxy)"),
]

# ── Demographic features (~29) ───────────────────────────────────────────────

_DEMOGRAPHIC: list[FeatureSpec] = [
    # Race / ethnicity
    FeatureSpec("pct_white_nh", "demographic", "race_ethnicity", "acs_tract", None, "Non-Hispanic white share"),
    FeatureSpec("pct_black", "demographic", "race_ethnicity", "acs_tract", None, "Black share"),
    FeatureSpec("pct_hispanic", "demographic", "race_ethnicity", "acs_tract", None, "Hispanic share"),
    FeatureSpec("pct_asian", "demographic", "race_ethnicity", "acs_tract", None, "Asian share"),

    # White working class
    FeatureSpec("pct_wwc", "demographic", "white_working_class", "acs_tract", None, "White non-Hispanic * (1 - BA+) interaction"),

    # Foreign born
    FeatureSpec("pct_foreign_born", "demographic", "foreign_born", "acs_tract", None, "Foreign-born share"),

    # Income
    FeatureSpec("median_hh_income", "demographic", "income", "acs_tract", None, "Median household income"),
    FeatureSpec("poverty_rate", "demographic", "income", "acs_tract", None, "Poverty rate"),
    FeatureSpec("gini", "demographic", "income", "acs_tract", None, "Gini coefficient"),

    # Education
    FeatureSpec("pct_ba_plus", "demographic", "education", "acs_tract", None, "Bachelor's degree or higher"),
    FeatureSpec("pct_graduate", "demographic", "education", "acs_tract", None, "Graduate degree share"),
    FeatureSpec("pct_no_hs", "demographic", "education", "acs_tract", None, "No high school diploma share"),

    # Housing
    FeatureSpec("pct_owner_occupied", "demographic", "housing", "acs_tract", None, "Owner-occupied share"),
    FeatureSpec("median_home_value", "demographic", "housing", "acs_tract", None, "Median home value"),
    FeatureSpec("pct_multi_unit", "demographic", "housing", "acs_tract", None, "Multi-unit housing share"),
    FeatureSpec("pct_pre_1960", "demographic", "housing", "acs_tract", None, "Pre-1960 housing share"),

    # Rent burden
    FeatureSpec("rent_burden", "demographic", "rent_burden", "acs_tract", None, "Annual rent / median income"),

    # Age and household
    FeatureSpec("median_age", "demographic", "age_household", "acs_tract", None, "Median age"),
    FeatureSpec("pct_under_18", "demographic", "age_household", "acs_tract", None, "Under 18 share"),
    FeatureSpec("pct_over_65", "demographic", "age_household", "acs_tract", None, "Over 65 share"),
    FeatureSpec("pct_single_hh", "demographic", "age_household", "acs_tract", None, "Single-person household share"),

    # Commute
    FeatureSpec("pct_wfh", "demographic", "commute", "acs_tract", None, "Work-from-home share"),
    FeatureSpec("mean_commute_time", "demographic", "commute", "acs_tract", None, "Mean commute time (minutes)"),
    FeatureSpec("pct_no_vehicle", "demographic", "commute", "acs_tract", None, "No-vehicle household share"),

    # Military
    FeatureSpec("pct_veteran", "demographic", "military", "acs_tract", None, "Veteran share"),
]

# ── Religion features (~4) ───────────────────────────────────────────────────

_RELIGION: list[FeatureSpec] = [
    FeatureSpec("evangelical_share", "religion", "religion", "rcms_county", None, "Evangelical Protestant adherent share"),
    FeatureSpec("catholic_share", "religion", "religion", "rcms_county", None, "Catholic adherent share"),
    FeatureSpec("black_protestant_share", "religion", "religion", "rcms_county", None, "Black Protestant adherent share"),
    FeatureSpec("adherence_rate", "religion", "religion", "rcms_county", None, "Total religious adherence rate"),
]

# ── Combined registry ────────────────────────────────────────────────────────

REGISTRY: list[FeatureSpec] = _ELECTORAL + _DEMOGRAPHIC + _RELIGION


def select_features(
    category: str | None = None,
    subcategory: str | None = None,
    exclude_year: int | None = None,
) -> list[str]:
    """Select feature names by category/subcategory, excluding features from a specific year.

    Parameters
    ----------
    category : str | None
        Filter to features in this category ("electoral", "demographic", "religion").
    subcategory : str | None
        Filter to features in this subcategory.
    exclude_year : int | None
        Exclude features whose source_year matches this year.

    Returns
    -------
    list[str]
        Feature column names matching the criteria.
    """
    result = []
    for spec in REGISTRY:
        if category is not None and spec.category != category:
            continue
        if subcategory is not None and spec.subcategory != subcategory:
            continue
        if exclude_year is not None and spec.source_year == exclude_year:
            continue
        result.append(spec.name)
    return result
