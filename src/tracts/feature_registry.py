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
    # Presidential shifts (log-odds, NOT state-centered — carry cross-state signal)
    FeatureSpec("pres_shift_2008_2012", "electoral", "presidential_shifts", "dra", 2012, "Dem log-odds shift 2008→2012"),  # noqa: E501
    FeatureSpec("pres_shift_2012_2016", "electoral", "presidential_shifts", "dra", 2016, "Dem log-odds shift 2012→2016"),  # noqa: E501
    FeatureSpec("pres_shift_2016_2020", "electoral", "presidential_shifts", "dra", 2020, "Dem log-odds shift 2016→2020"),  # noqa: E501
    FeatureSpec("pres_shift_2020_2024", "electoral", "presidential_shifts", "dra", 2024, "Dem log-odds shift 2020→2024"),  # noqa: E501
    FeatureSpec("pres_turnout_shift_2016_2020", "electoral", "presidential_shifts", "dra", 2020, "Turnout proportional change 2016→2020"),  # noqa: E501
    FeatureSpec("pres_turnout_shift_2020_2024", "electoral", "presidential_shifts", "dra", 2024, "Turnout proportional change 2020→2024"),  # noqa: E501

    # Governor shifts (log-odds, state-centered — proxy for candidate effect removal)
    FeatureSpec("gov_shift_2010_2014", "electoral", "governor_shifts", "dra", 2014, "Gov Dem shift 2010→2014 (state-centered)"),  # noqa: E501
    FeatureSpec("gov_shift_2014_2018", "electoral", "governor_shifts", "dra", 2018, "Gov Dem shift 2014→2018 (state-centered)"),  # noqa: E501
    FeatureSpec("gov_shift_2018_2022", "electoral", "governor_shifts", "dra", 2022, "Gov Dem shift 2018→2022 (state-centered)"),  # noqa: E501

    # Senate shifts (log-odds, state-centered — proxy for candidate effect removal)
    FeatureSpec("sen_shift_2010_2014", "electoral", "senate_shifts", "dra", 2014, "Senate Dem shift 2010→2014 (state-centered)"),  # noqa: E501
    FeatureSpec("sen_shift_2014_2016", "electoral", "senate_shifts", "dra", 2016, "Senate Dem shift 2014→2016 (state-centered)"),  # noqa: E501
    FeatureSpec("sen_shift_2016_2020", "electoral", "senate_shifts", "dra", 2020, "Senate Dem shift 2016→2020 (state-centered)"),  # noqa: E501
    FeatureSpec("sen_shift_2020_2022", "electoral", "senate_shifts", "dra", 2022, "Senate Dem shift 2020→2022 (state-centered)"),  # noqa: E501

    # House shifts (log-odds, state-centered)
    FeatureSpec("house_shift_2016_2018", "electoral", "house_shifts", "dra", 2018, "House Dem shift 2016→2018 (state-centered)"),  # noqa: E501
    FeatureSpec("house_shift_2018_2020", "electoral", "house_shifts", "dra", 2020, "House Dem shift 2018→2020 (state-centered)"),  # noqa: E501

    # Presidential lean (dem_share levels)
    FeatureSpec("pres_dem_share_2016", "electoral", "presidential_lean", "dra", 2016, "Dem vote share 2016 president"),
    FeatureSpec("pres_dem_share_2020", "electoral", "presidential_lean", "dra", 2020, "Dem vote share 2020 president"),
    FeatureSpec("pres_dem_share_2024", "electoral", "presidential_lean", "dra", 2024, "Dem vote share 2024 president"),

    # Turnout level (raw total votes)
    FeatureSpec("turnout_2016", "electoral", "turnout_level", "dra", 2016, "Total votes cast 2016 president"),
    FeatureSpec("turnout_2020", "electoral", "turnout_level", "dra", 2020, "Total votes cast 2020 president"),
    FeatureSpec("turnout_2024", "electoral", "turnout_level", "dra", 2024, "Total votes cast 2024 president"),

    # Turnout shift (proportional change in total votes)
    FeatureSpec("turnout_shift_2016_2020", "electoral", "turnout_shift", "dra", 2020, "Turnout proportional change 2016→2020"),  # noqa: E501
    FeatureSpec("turnout_shift_2020_2024", "electoral", "turnout_shift", "dra", 2024, "Turnout proportional change 2020→2024"),  # noqa: E501

    # Vote density (votes / sq km)
    FeatureSpec("vote_density_2020", "electoral", "vote_density", "dra", 2020, "Votes per sq km 2020"),
    FeatureSpec("vote_density_2024", "electoral", "vote_density", "dra", 2024, "Votes per sq km 2024"),

    # Split ticket
    FeatureSpec("split_ticket_2016", "electoral", "split_ticket", "dra", 2016, "abs(pres_dem - house_dem) 2016"),
    FeatureSpec("split_ticket_2020", "electoral", "split_ticket", "dra", 2020, "abs(pres_dem - house_dem) 2020"),

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
    FeatureSpec("pct_wwc", "demographic", "white_working_class", "acs_tract", None, "White non-Hispanic * (1 - BA+) interaction"),  # noqa: E501

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
    FeatureSpec("evangelical_share", "religion", "religion", "rcms_county", None, "Evangelical Protestant adherent share"),  # noqa: E501
    FeatureSpec("catholic_share", "religion", "religion", "rcms_county", None, "Catholic adherent share"),
    FeatureSpec("black_protestant_share", "religion", "religion", "rcms_county", None, "Black Protestant adherent share"),  # noqa: E501
    FeatureSpec("adherence_rate", "religion", "religion", "rcms_county", None, "Total religious adherence rate"),
    FeatureSpec("lds_rate", "religion", "religion", "rcms_county", None, "LDS/Mormon adherents per 1k"),
    FeatureSpec("muslim_rate", "religion", "religion", "rcms_county", None, "Muslim adherents per 1k"),
    FeatureSpec("jewish_rate", "religion", "religion", "rcms_county", None, "Jewish adherents per 1k"),
    FeatureSpec("hindu_sikh_rate", "religion", "religion", "rcms_county", None, "Hindu+Sikh proxy per 1k"),
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
