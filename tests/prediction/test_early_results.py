"""Tests for early-cycle election results integration.

Early results (special elections, off-year governor/SC races) are injected
as high-confidence poll-like observations with time decay.  These tests verify:
  - CSV loading behaves correctly for valid, empty, and missing files
  - merge_early_results handles all edge cases cleanly
  - n_sample capping prevents any single result from dominating
  - Generic Ballot entries are correctly extracted for compute_gb_shift()
  - The pipeline integration produces the expected directional shifts

Each test targets behavior, not implementation internals.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.prediction.early_results import (
    _GB_GEO_LEVEL,
    _GB_RACE_LABEL,
    extract_gb_observations,
    load_early_results,
    merge_early_results,
)
from src.prediction.generic_ballot import compute_gb_shift

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a polls-format CSV to ``path``."""
    if not rows:
        path.write_text("race,geography,geo_level,dem_share,n_sample,date,pollster,notes\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def tmp_early_results(tmp_path: Path) -> Path:
    """Write a minimal early_cycle_results.csv and return its path."""
    csv_path = tmp_path / "early_cycle_results.csv"
    rows = [
        {
            "race": _GB_RACE_LABEL,
            "geography": "US",
            "geo_level": _GB_GEO_LEVEL,
            "dem_share": "0.572",
            "n_sample": "5000",
            "date": "2025-11-04",
            "pollster": "VA Governor Result",
            "notes": "VA 2025 Governor",
        },
        {
            "race": _GB_RACE_LABEL,
            "geography": "US",
            "geo_level": _GB_GEO_LEVEL,
            "dem_share": "0.56",
            "n_sample": "5000",
            "date": "2025-11-04",
            "pollster": "NJ Governor Result",
            "notes": "NJ 2025 Governor",
        },
        {
            "race": "2026 WI Governor",
            "geography": "WI",
            "geo_level": "state",
            "dem_share": "0.55",
            "n_sample": "5000",
            "date": "2026-04-07",
            "pollster": "WI Supreme Court Result",
            "notes": "WI SC adjusted",
        },
        {
            "race": "2026 GA Senate",
            "geography": "GA",
            "geo_level": "state",
            "dem_share": "0.545",
            "n_sample": "5000",
            "date": "2026-04-08",
            "pollster": "GA-14 Special Result",
            "notes": "GA-14 district-to-state adjusted",
        },
    ]
    _write_csv(csv_path, rows)
    return csv_path


# ---------------------------------------------------------------------------
# load_early_results tests
# ---------------------------------------------------------------------------


def test_load_early_results_valid_file(tmp_early_results: Path) -> None:
    """Loading a valid CSV returns all 4 rows grouped by race."""
    result = load_early_results(path=tmp_early_results)

    assert _GB_RACE_LABEL in result
    assert "2026 WI Governor" in result
    assert "2026 GA Senate" in result

    # Two GB entries
    assert len(result[_GB_RACE_LABEL]) == 2
    # One WI entry
    assert len(result["2026 WI Governor"]) == 1
    # One GA entry
    assert len(result["2026 GA Senate"]) == 1


def test_load_early_results_poll_dict_keys(tmp_early_results: Path) -> None:
    """Each poll dict must contain the fields prepare_polls() expects."""
    result = load_early_results(path=tmp_early_results)
    for race_polls in result.values():
        for poll in race_polls:
            assert "dem_share" in poll
            assert "n_sample" in poll
            assert "state" in poll  # geography mapped to state key
            assert "date" in poll


def test_load_early_results_empty_file(tmp_path: Path) -> None:
    """An empty CSV (header only) returns an empty dict, not an error."""
    csv_path = tmp_path / "empty.csv"
    _write_csv(csv_path, [])
    result = load_early_results(path=csv_path)
    assert result == {}


def test_load_early_results_missing_file(tmp_path: Path) -> None:
    """A missing CSV returns an empty dict with a warning, not an exception."""
    missing = tmp_path / "does_not_exist.csv"
    result = load_early_results(path=missing)
    assert result == {}


def test_load_early_results_skips_malformed_rows(tmp_path: Path) -> None:
    """Rows with non-numeric dem_share are skipped, valid rows are kept."""
    csv_path = tmp_path / "malformed.csv"
    rows = [
        {
            "race": "2026 WI Governor",
            "geography": "WI",
            "geo_level": "state",
            "dem_share": "NOT_A_NUMBER",
            "n_sample": "500",
            "date": "2026-04-07",
            "pollster": "Test",
            "notes": "",
        },
        {
            "race": "2026 GA Senate",
            "geography": "GA",
            "geo_level": "state",
            "dem_share": "0.545",
            "n_sample": "500",
            "date": "2026-04-08",
            "pollster": "Test",
            "notes": "",
        },
    ]
    _write_csv(csv_path, rows)
    result = load_early_results(path=csv_path)
    # WI row is malformed; GA row is valid
    assert "2026 WI Governor" not in result
    assert "2026 GA Senate" in result


def test_load_early_results_n_sample_capping(tmp_path: Path) -> None:
    """n_sample values above max_effective_n are silently capped."""
    csv_path = tmp_path / "big_n.csv"
    rows = [
        {
            "race": "2026 WI Governor",
            "geography": "WI",
            "geo_level": "state",
            "dem_share": "0.55",
            "n_sample": "999999",  # Way above the cap
            "date": "2026-04-07",
            "pollster": "Test",
            "notes": "",
        },
    ]
    _write_csv(csv_path, rows)
    result = load_early_results(path=csv_path)
    assert result["2026 WI Governor"][0]["n_sample"] <= 5000


def test_load_early_results_respects_custom_max_n(tmp_path: Path) -> None:
    """max_effective_n from config file is respected."""
    csv_path = tmp_path / "polls.csv"
    rows = [
        {
            "race": "2026 WI Governor",
            "geography": "WI",
            "geo_level": "state",
            "dem_share": "0.55",
            "n_sample": "10000",
            "date": "2026-04-07",
            "pollster": "Test",
            "notes": "",
        },
    ]
    _write_csv(csv_path, rows)

    # Write a params config with a low max_n
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"early_results": {"enabled": True, "max_effective_n": 200}}))

    result = load_early_results(path=csv_path, params_path=params_path)
    assert result["2026 WI Governor"][0]["n_sample"] == 200


def test_load_early_results_disabled_via_config(tmp_early_results: Path, tmp_path: Path) -> None:
    """When early_results.enabled=false in config, returns empty dict."""
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"early_results": {"enabled": False}}))

    result = load_early_results(path=tmp_early_results, params_path=params_path)
    assert result == {}


# ---------------------------------------------------------------------------
# merge_early_results tests
# ---------------------------------------------------------------------------


def test_merge_early_results_no_overlap() -> None:
    """Races present only in early results are added correctly."""
    polls_by_race = {
        "2026 FL Governor": [{"dem_share": 0.48, "n_sample": 600, "state": "FL", "geo_level": "state"}],
    }
    early = {
        "2026 WI Governor": [{"dem_share": 0.55, "n_sample": 5000, "state": "WI", "geo_level": "state"}],
    }
    merged = merge_early_results(polls_by_race, early)

    assert "2026 FL Governor" in merged
    assert "2026 WI Governor" in merged
    assert len(merged["2026 WI Governor"]) == 1


def test_merge_early_results_overlapping_races() -> None:
    """Early results for an existing race append to existing polls."""
    polls_by_race = {
        "2026 GA Senate": [{"dem_share": 0.52, "n_sample": 800, "state": "GA", "geo_level": "state"}],
    }
    early = {
        "2026 GA Senate": [{"dem_share": 0.545, "n_sample": 5000, "state": "GA", "geo_level": "state"}],
    }
    merged = merge_early_results(polls_by_race, early)

    # Should have original poll + early result
    assert len(merged["2026 GA Senate"]) == 2


def test_merge_early_results_empty_early() -> None:
    """Empty early results return the original polls unchanged."""
    polls_by_race = {"2026 FL Governor": [{"dem_share": 0.48, "n_sample": 600, "state": "FL"}]}
    merged = merge_early_results(polls_by_race, {})
    assert merged == polls_by_race


def test_merge_early_results_gb_entries_excluded() -> None:
    """Generic Ballot entries (geo_level=national) are NOT added to polls_by_race.

    They must go through compute_gb_shift() via extra_gb_polls instead,
    to avoid double-counting with the file-based GB loading.
    """
    polls_by_race: dict = {}
    early = {
        _GB_RACE_LABEL: [
            {"dem_share": 0.572, "n_sample": 5000, "state": "US", "geo_level": _GB_GEO_LEVEL},
        ],
    }
    merged = merge_early_results(polls_by_race, early)
    # GB entry should not appear in merged polls_by_race
    assert _GB_RACE_LABEL not in merged


def test_merge_early_results_does_not_mutate_inputs() -> None:
    """merge_early_results must not modify the input dicts."""
    polls_by_race = {"2026 GA Senate": [{"dem_share": 0.52, "n_sample": 800}]}
    original_polls = dict(polls_by_race)
    early = {"2026 GA Senate": [{"dem_share": 0.545, "n_sample": 5000}]}
    merge_early_results(polls_by_race, early)
    assert polls_by_race == original_polls


# ---------------------------------------------------------------------------
# extract_gb_observations tests
# ---------------------------------------------------------------------------


def test_extract_gb_observations_returns_tuples(tmp_early_results: Path) -> None:
    """GB observations are (dem_share, n_sample) tuples with correct types."""
    early = load_early_results(path=tmp_early_results)
    gb_obs = extract_gb_observations(early)

    assert len(gb_obs) == 2  # VA + NJ entries
    for dem_share, n_sample in gb_obs:
        assert isinstance(dem_share, float)
        assert isinstance(n_sample, int)
        assert 0.0 < dem_share < 1.0
        assert n_sample > 0


def test_extract_gb_observations_only_national(tmp_early_results: Path) -> None:
    """Only entries with geo_level=national are extracted as GB observations."""
    early = load_early_results(path=tmp_early_results)
    gb_obs = extract_gb_observations(early)

    # VA + NJ = 2 entries; WI and GA are state-level so excluded
    assert len(gb_obs) == 2
    dem_shares = {round(d, 3) for d, _ in gb_obs}
    assert 0.572 in dem_shares
    assert 0.56 in dem_shares


def test_extract_gb_observations_empty_dict() -> None:
    """Empty early results dict returns empty list."""
    result = extract_gb_observations({})
    assert result == []


def test_extract_gb_observations_no_gb_entries() -> None:
    """Dict with only state-level entries returns empty list."""
    early = {
        "2026 WI Governor": [{"dem_share": 0.55, "n_sample": 5000, "geo_level": "state"}],
    }
    result = extract_gb_observations(early)
    assert result == []


# ---------------------------------------------------------------------------
# Integration: extra_gb_polls in compute_gb_shift
# ---------------------------------------------------------------------------


def test_compute_gb_shift_extra_polls_increase_shift(tmp_path: Path) -> None:
    """Injecting strong D-leaning extra polls increases the GB shift."""
    # Write a minimal polls CSV so compute_gb_shift doesn't depend on real data
    polls_csv = tmp_path / "polls.csv"
    rows = [
        {
            "race": _GB_RACE_LABEL,
            "geography": "US",
            "geo_level": _GB_GEO_LEVEL,
            "dem_share": "0.52",
            "n_sample": "1000",
            "date": "2026-01-01",
            "pollster": "Test Pollster",
            "notes": "",
        },
    ]
    _write_csv(polls_csv, rows)

    # Baseline: no extra polls
    baseline = compute_gb_shift(
        polls_path=polls_csv,
        yougov_json_path=tmp_path / "no_yougov.json",  # non-existent → empty
    )

    # With strong D-leaning extra polls (VA/NJ governor-level D margins)
    extra = [(0.572, 5000), (0.56, 5000)]
    with_extra = compute_gb_shift(
        polls_path=polls_csv,
        yougov_json_path=tmp_path / "no_yougov.json",
        extra_gb_polls=extra,
    )

    # GB shift should be higher (more D) when extra D-leaning results are added
    assert with_extra.shift > baseline.shift


def test_compute_gb_shift_no_extra_polls_unchanged(tmp_path: Path) -> None:
    """Passing extra_gb_polls=None produces same result as no parameter."""
    polls_csv = tmp_path / "polls.csv"
    rows = [
        {
            "race": _GB_RACE_LABEL,
            "geography": "US",
            "geo_level": _GB_GEO_LEVEL,
            "dem_share": "0.52",
            "n_sample": "1000",
            "date": "2026-01-01",
            "pollster": "Test",
            "notes": "",
        },
    ]
    _write_csv(polls_csv, rows)

    no_extra = compute_gb_shift(polls_path=polls_csv, yougov_json_path=tmp_path / "x.json")
    with_none = compute_gb_shift(
        polls_path=polls_csv,
        yougov_json_path=tmp_path / "x.json",
        extra_gb_polls=None,
    )
    assert no_extra.shift == pytest.approx(with_none.shift)


# ---------------------------------------------------------------------------
# Integration with prepare_polls (time decay end-to-end)
# ---------------------------------------------------------------------------


def test_early_results_flow_through_prepare_polls(tmp_early_results: Path) -> None:
    """State-level early results flow through prepare_polls and receive time decay."""
    from src.prediction.forecast_engine import prepare_polls

    early = load_early_results(path=tmp_early_results)
    # Simulate merge: use only the state-level early results
    polls_by_race: dict = {}
    merged = merge_early_results(polls_by_race, early)

    # WI and GA entries should be present in merged
    assert "2026 WI Governor" in merged
    assert "2026 GA Senate" in merged

    # Run through prepare_polls with a reference date far in the future
    # to verify time decay is applied (n_sample should be reduced)
    reference_date = "2026-11-03"  # election day, months after election date
    weighted = prepare_polls(
        merged,
        reference_date=reference_date,
        half_life_days=30.0,
    )

    # After 6+ months of decay, n_sample for WI (2026-04-07) should be << 5000
    wi_polls = weighted.get("2026 WI Governor", [])
    assert len(wi_polls) == 1
    # ~7 months = ~210 days; at half_life=30 days, decay ≈ 2^(-7) ≈ 0.0078
    # So n_effective ≈ 5000 * 0.0078 ≈ 39.  Assert substantial reduction.
    assert wi_polls[0]["n_sample"] < 500  # well below original 5000
