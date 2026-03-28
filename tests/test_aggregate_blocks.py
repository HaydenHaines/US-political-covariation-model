"""Tests for DRA block-to-tract aggregation."""
import pandas as pd
import numpy as np
import pytest
from src.tracts.aggregate_blocks_to_tracts import detect_race_columns


def test_detect_race_columns_finds_standard():
    cols = ["GEOID", "E_08_PRES_Total", "E_08_PRES_Dem", "E_08_PRES_Rep",
            "E_20_PRES_Total", "E_20_PRES_Dem", "E_20_PRES_Rep"]
    result = detect_race_columns(cols)
    assert ("E_08_PRES", 2008, "president") in result
    assert ("E_20_PRES", 2020, "president") in result


def test_detect_race_columns_finds_governor():
    cols = ["GEOID", "E_14_GOV_Total", "E_14_GOV_Dem", "E_14_GOV_Rep",
            "E_18_GOV_Total", "E_18_GOV_Dem", "E_18_GOV_Rep"]
    result = detect_race_columns(cols)
    assert ("E_14_GOV", 2014, "governor") in result
    assert ("E_18_GOV", 2018, "governor") in result


def test_detect_race_columns_finds_senate():
    cols = ["GEOID", "E_16_SEN_Total", "E_16_SEN_Dem", "E_16_SEN_Rep",
            "E_24_SEN_Total", "E_24_SEN_Dem", "E_24_SEN_Rep"]
    result = detect_race_columns(cols)
    assert ("E_16_SEN", 2016, "senate") in result
    assert ("E_24_SEN", 2024, "senate") in result


def test_detect_race_columns_finds_senate_special():
    cols = ["GEOID", "E_20_SEN_SPEC_Total", "E_20_SEN_SPEC_Dem", "E_20_SEN_SPEC_Rep"]
    result = detect_race_columns(cols)
    assert any(r[2] == "senate" and r[1] == 2020 for r in result)
    assert ("E_20_SEN_SPEC", 2020, "senate") in result


def test_detect_race_columns_finds_senate_runoff():
    cols = ["GEOID", "E_20_SEN_ROFF_Total", "E_20_SEN_ROFF_Dem", "E_20_SEN_ROFF_Rep"]
    result = detect_race_columns(cols)
    assert ("E_20_SEN_ROFF", 2020, "senate") in result


def test_detect_race_columns_finds_senate_special_runoff():
    cols = ["GEOID", "E_20_SEN_SPECROFF_Total", "E_20_SEN_SPECROFF_Dem", "E_20_SEN_SPECROFF_Rep"]
    result = detect_race_columns(cols)
    assert ("E_20_SEN_SPECROFF", 2020, "senate") in result


def test_detect_race_columns_skips_comp():
    """COMP (composite) columns should be excluded — they're synthetic averages."""
    cols = ["GEOID", "E_16-20_COMP_Total", "E_16-20_COMP_Dem", "E_16-20_COMP_Rep"]
    result = detect_race_columns(cols)
    assert len(result) == 0


def test_detect_race_columns_skips_ag_and_cong():
    """AG and CONG columns should be excluded — not presidential/governor/senate."""
    cols = ["GEOID", "E_18_AG_Total", "E_18_AG_Dem", "E_18_AG_Rep",
            "E_22_CONG_Total", "E_22_CONG_Dem", "E_22_CONG_Rep"]
    result = detect_race_columns(cols)
    assert len(result) == 0


def test_detect_race_columns_requires_dem_col():
    """If _Dem column is missing, the race should be skipped."""
    cols = ["GEOID", "E_20_PRES_Total", "E_20_PRES_Rep"]
    result = detect_race_columns(cols)
    assert len(result) == 0


def test_detect_race_columns_mixed():
    """Realistic mix of columns from a state with many races."""
    cols = [
        "GEOID",
        "E_08_PRES_Total", "E_08_PRES_Dem", "E_08_PRES_Rep",
        "E_12_PRES_Total", "E_12_PRES_Dem", "E_12_PRES_Rep",
        "E_14_GOV_Total", "E_14_GOV_Dem", "E_14_GOV_Rep",
        "E_16_SEN_Total", "E_16_SEN_Dem", "E_16_SEN_Rep",
        "E_20_SEN_SPEC_Total", "E_20_SEN_SPEC_Dem", "E_20_SEN_SPEC_Rep",
        "E_16-20_COMP_Total", "E_16-20_COMP_Dem", "E_16-20_COMP_Rep",
        "E_18_AG_Total", "E_18_AG_Dem", "E_18_AG_Rep",
        "POP_Total",
    ]
    result = detect_race_columns(cols)
    prefixes = {r[0] for r in result}
    assert prefixes == {"E_08_PRES", "E_12_PRES", "E_14_GOV", "E_16_SEN", "E_20_SEN_SPEC"}
    assert len(result) == 5
