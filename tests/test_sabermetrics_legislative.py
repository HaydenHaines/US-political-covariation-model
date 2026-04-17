"""Tests for legislative.py: VoteView + LES loading and career summary building.

All tests use synthetic data (no filesystem access) except for the integration
tests that read already-downloaded files and the saved parquet. The integration
tests are skipped if the data files are absent.

Coverage:
  1. load_voteview() — filtering, column selection, congress cutoff
  2. _load_les_house() / _load_les_senate() via load_les()
  3. build_legislative_stats() — matching, career means, boolean flag
  4. _safe_float() — edge cases
  5. _validate_les_column() — detects shifted columns
  6. Integration: parquet exists and has correct shape/types
  7. Integration: key candidate sample values are in plausible ranges
  8. Null handling — candidates with no legislative record
  9. Multi-congress career mean computation
  10. ICPSR bridge — LES joins to VoteView via icpsr
  11. download_voteview() / download_les() skip if files exist
  12. bioguide_id in registry key (not inner field) is used correctly
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.sabermetrics.legislative import (
    _safe_float,
    _validate_les_column,
    build_legislative_stats,
    load_voteview,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VV_FILE = PROJECT_ROOT / "data" / "raw" / "voteview" / "HSall_members.csv"
LES_DIR = PROJECT_ROOT / "data" / "raw" / "les"
PARQUET_FILE = PROJECT_ROOT / "data" / "sabermetrics" / "legislative_stats.parquet"


# ---------------------------------------------------------------------------
# Fixtures: synthetic VoteView data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_voteview_df() -> pd.DataFrame:
    """Synthetic VoteView-shaped DataFrame with 6 member-congress rows."""
    return pd.DataFrame({
        "bioguide_id": ["A000001", "A000001", "B000002", "B000002", "C000003", "C000003"],
        "congress": [100, 101, 100, 101, 110, 111],
        "chamber": ["Senate", "Senate", "House", "House", "Senate", "Senate"],
        "icpsr": [10001, 10001, 10002, 10002, 10003, 10003],
        "nominate_dim1": [-0.5, -0.5, 0.3, 0.3, -0.1, -0.1],
        "nokken_poole_dim1": [-0.52, -0.48, 0.28, 0.32, -0.11, -0.09],
    })


@pytest.fixture
def synthetic_les_df() -> pd.DataFrame:
    """Synthetic LES-shaped DataFrame with matching bioguide_ids."""
    return pd.DataFrame({
        "bioguide_id": ["A000001", "A000001", "B000002"],
        "congress": [100, 101, 100],
        "chamber": ["Senate", "Senate", "House"],
        "les_score": [1.2, 1.4, 0.8],
        "les2_score": [1.0, 1.6, 0.9],
    })


@pytest.fixture
def minimal_registry() -> dict:
    """Registry with three candidates: two with VoteView + LES records, one with none."""
    return {
        "persons": {
            "A000001": {"name": "Alice Senator", "party": "D", "bioguide_id": "A000001", "races": []},
            "B000002": {"name": "Bob Rep", "party": "R", "bioguide_id": "B000002", "races": []},
            "Z999999": {"name": "Zara Newcomer", "party": "D", "bioguide_id": "Z999999", "races": []},
        }
    }


# ---------------------------------------------------------------------------
# 1. load_voteview() — filtering
# ---------------------------------------------------------------------------

def test_load_voteview_drops_pre_93rd_congress(tmp_path: Path) -> None:
    """load_voteview() must exclude congresses before the 93rd (< 1973)."""
    csv_file = tmp_path / "HSall_members.csv"
    rows = pd.DataFrame({
        "bioguide_id": ["A000001", "A000001", "B000002"],
        "congress": [92, 93, 100],   # 92nd should be dropped
        "chamber": ["Senate", "Senate", "House"],
        "icpsr": [1, 1, 2],
        "nominate_dim1": [-0.3, -0.3, 0.5],
        "nokken_poole_dim1": [-0.31, -0.29, 0.51],
    })
    rows.to_csv(csv_file, index=False)
    result = load_voteview(csv_file)
    assert (result["congress"] >= 93).all(), "Pre-93rd congresses should be dropped"
    assert 92 not in result["congress"].values


def test_load_voteview_drops_rows_without_bioguide(tmp_path: Path) -> None:
    """load_voteview() must exclude rows with no bioguide_id."""
    csv_file = tmp_path / "HSall_members.csv"
    rows = pd.DataFrame({
        "bioguide_id": ["A000001", None, "C000003"],
        "congress": [100, 100, 100],
        "chamber": ["Senate", "House", "Senate"],
        "icpsr": [1, 2, 3],
        "nominate_dim1": [-0.3, 0.5, -0.1],
        "nokken_poole_dim1": [-0.31, 0.51, -0.09],
    })
    rows.to_csv(csv_file, index=False)
    result = load_voteview(csv_file)
    assert result["bioguide_id"].notna().all(), "Rows with null bioguide_id should be excluded"
    assert len(result) == 2


def test_load_voteview_output_columns(tmp_path: Path) -> None:
    """load_voteview() must return the required columns."""
    csv_file = tmp_path / "HSall_members.csv"
    rows = pd.DataFrame({
        "bioguide_id": ["A000001"],
        "congress": [100],
        "chamber": ["Senate"],
        "icpsr": [1],
        "nominate_dim1": [-0.3],
        "nokken_poole_dim1": [-0.31],
    })
    rows.to_csv(csv_file, index=False)
    result = load_voteview(csv_file)
    required = {"bioguide_id", "congress", "chamber", "icpsr", "nominate_dim1", "nokken_poole_dim1"}
    assert required.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# 2. build_legislative_stats() — core logic
# ---------------------------------------------------------------------------

def test_build_stats_career_means(
    minimal_registry: dict,
    synthetic_voteview_df: pd.DataFrame,
    synthetic_les_df: pd.DataFrame,
) -> None:
    """Career means should average across all congresses for the member."""
    stats = build_legislative_stats(minimal_registry, synthetic_voteview_df, synthetic_les_df)
    # A000001 has two congress rows: nominate_dim1 = [-0.5, -0.5] → mean = -0.5
    assert abs(stats.loc["A000001", "career_nominate_dim1"] - (-0.5)) < 1e-9
    # A000001 nokken_poole = [-0.52, -0.48] → mean = -0.5
    assert abs(stats.loc["A000001", "career_nokken_poole_dim1"] - (-0.5)) < 1e-9
    # A000001 les = [1.2, 1.4] → mean = 1.3
    assert abs(stats.loc["A000001", "career_les"] - 1.3) < 1e-9


def test_build_stats_no_record_candidate(
    minimal_registry: dict,
    synthetic_voteview_df: pd.DataFrame,
    synthetic_les_df: pd.DataFrame,
) -> None:
    """Candidate with no VoteView or LES match should have has_legislative_record=False."""
    stats = build_legislative_stats(minimal_registry, synthetic_voteview_df, synthetic_les_df)
    row = stats.loc["Z999999"]
    assert row["has_legislative_record"] is False or row["has_legislative_record"] == False  # noqa: E712
    assert math.isnan(row["career_nominate_dim1"])
    assert math.isnan(row["career_les"])
    assert row["congresses_served_vv"] == 0
    assert row["congresses_served_les"] == 0


def test_build_stats_vv_match_no_les_match(
    minimal_registry: dict,
    synthetic_voteview_df: pd.DataFrame,
) -> None:
    """Candidate with VoteView match but no LES match should have has_legislative_record=True."""
    # C000003 is in VoteView but not in LES — add to registry
    registry = {
        "persons": {
            "C000003": {"name": "Charlie Senator", "party": "R", "bioguide_id": "C000003", "races": []},
        }
    }
    empty_les = pd.DataFrame(columns=["bioguide_id", "congress", "chamber", "les_score", "les2_score"])
    stats = build_legislative_stats(registry, synthetic_voteview_df, empty_les)
    row = stats.loc["C000003"]
    assert row["has_legislative_record"] is True or row["has_legislative_record"] == True  # noqa: E712
    assert not math.isnan(row["career_nominate_dim1"])
    assert math.isnan(row["career_les"])


def test_build_stats_index_is_person_id(
    minimal_registry: dict,
    synthetic_voteview_df: pd.DataFrame,
    synthetic_les_df: pd.DataFrame,
) -> None:
    """Output DataFrame index should be person_id strings from the registry."""
    stats = build_legislative_stats(minimal_registry, synthetic_voteview_df, synthetic_les_df)
    assert stats.index.name == "person_id"
    assert set(stats.index) == {"A000001", "B000002", "Z999999"}


def test_build_stats_congresses_served_count(
    minimal_registry: dict,
    synthetic_voteview_df: pd.DataFrame,
    synthetic_les_df: pd.DataFrame,
) -> None:
    """congresses_served_vv should count congress-level rows in VoteView."""
    stats = build_legislative_stats(minimal_registry, synthetic_voteview_df, synthetic_les_df)
    # A000001 has 2 congress rows
    assert stats.loc["A000001", "congresses_served_vv"] == 2
    # B000002 has 2 congress rows
    assert stats.loc["B000002", "congresses_served_vv"] == 2
    # LES: A000001 has 2 rows
    assert stats.loc["A000001", "congresses_served_les"] == 2
    # LES: B000002 has 1 row
    assert stats.loc["B000002", "congresses_served_les"] == 1


# ---------------------------------------------------------------------------
# 3. _safe_float() — edge cases
# ---------------------------------------------------------------------------

def test_safe_float_valid_number() -> None:
    assert _safe_float(1.5) == 1.5


def test_safe_float_string_number() -> None:
    assert _safe_float("3.14") == pytest.approx(3.14)


def test_safe_float_none_returns_nan() -> None:
    assert math.isnan(_safe_float(None))


def test_safe_float_empty_string_returns_nan() -> None:
    assert math.isnan(_safe_float(""))


def test_safe_float_nan_input_returns_nan() -> None:
    assert math.isnan(_safe_float(float("nan")))


# ---------------------------------------------------------------------------
# 4. _validate_les_column() — column position validation
# ---------------------------------------------------------------------------

def test_validate_les_column_passes_on_match() -> None:
    """No exception when column name contains the expected keyword."""
    df = pd.DataFrame({"icpsr number": [1], "congress number": [93], "LES Classic": [1.0]})
    _validate_les_column(df, 0, "icpsr", "Senate")  # should not raise


def test_validate_les_column_raises_on_mismatch() -> None:
    """ValueError when column name does not contain expected keyword."""
    df = pd.DataFrame({"wrong_col": [1], "congress number": [93]})
    with pytest.raises(ValueError, match="icpsr"):
        _validate_les_column(df, 0, "icpsr", "House")


# ---------------------------------------------------------------------------
# 5. download functions skip when files already exist
# ---------------------------------------------------------------------------

def test_download_voteview_skips_if_file_exists(tmp_path: Path) -> None:
    """download_voteview() should not re-download if file is already present."""
    from src.sabermetrics.legislative import download_voteview

    dest_file = tmp_path / "HSall_members.csv"
    dest_file.write_text("already_present")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        result = download_voteview(tmp_path)
        mock_retrieve.assert_not_called()
    assert result == dest_file


def test_download_les_skips_if_files_exist(tmp_path: Path) -> None:
    """download_les() should not re-download if both files are already present."""
    from src.sabermetrics.legislative import download_les

    house_file = tmp_path / "CELHouse93to118.xlsx"
    senate_file = tmp_path / "CELSenate93to118.xls"
    house_file.write_text("house_placeholder")
    senate_file.write_text("senate_placeholder")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        result = download_les(tmp_path)
        mock_retrieve.assert_not_called()
    assert result == (house_file, senate_file)


# ---------------------------------------------------------------------------
# 6. ICPSR bridge test — LES joins via ICPSR from VoteView
# ---------------------------------------------------------------------------

def test_les_icpsr_bridge_matches_correct_bioguide(
    synthetic_voteview_df: pd.DataFrame,
) -> None:
    """LES records linked via ICPSR should resolve to the correct bioguide_id."""
    from src.sabermetrics.legislative import load_les

    with tempfile.TemporaryDirectory() as tmpdir:
        les_dir = Path(tmpdir)
        # Create minimal House Excel that mimics real column layout
        # We use a DataFrame with the correct column count (88 cols), but only
        # the key positions matter for the loader.
        n_cols = 88
        col_names = [f"col_{i}" for i in range(n_cols)]
        # Set the actual meaningful names at the expected positions
        col_names[1] = "Legislator name, as given in THOMAS"
        col_names[2] = "ICPSR number, according to Poole and Rosenthal"
        col_names[3] = "Congress number"
        col_names[52] = "LES 1.0"
        col_names[68] = "LES 2.0"

        row_data = {c: [None] for c in col_names}
        row_data[col_names[2]] = [10001]   # ICPSR for A000001
        row_data[col_names[3]] = [100]
        row_data[col_names[52]] = [2.5]
        row_data[col_names[68]] = [3.0]

        house_df = pd.DataFrame(row_data)
        house_path = les_dir / "CELHouse93to118.xlsx"
        house_df.to_excel(house_path, index=False)

        # Create minimal Senate Excel (same approach)
        n_cols_s = 88
        col_names_s = [f"scol_{i}" for i in range(n_cols_s)]
        col_names_s[0] = "last name"
        col_names_s[1] = "first name"
        col_names_s[3] = "congress number"
        col_names_s[5] = "icpsr number"
        col_names_s[50] = "LES Classic, not including"
        col_names_s[66] = "LES 2.0"

        senate_data = {c: [None] for c in col_names_s}
        senate_data[col_names_s[5]] = [10002]   # ICPSR for B000002
        senate_data[col_names_s[3]] = [100]
        senate_data[col_names_s[50]] = [1.1]
        senate_data[col_names_s[66]] = [1.2]

        senate_df = pd.DataFrame(senate_data)
        senate_path = les_dir / "CELSenate93to118.xls"
        senate_df.to_excel(senate_path, index=False, engine="xlwt" if False else "openpyxl")
        # xlwt not available — use openpyxl and save as xlsx, then rename
        import shutil
        senate_xlsx = les_dir / "CELSenate93to118.xlsx"
        senate_df.to_excel(senate_xlsx, index=False)
        shutil.copy(senate_xlsx, senate_path)

        result = load_les(les_dir, synthetic_voteview_df)
        # ICPSR 10001 → bioguide A000001 (House)
        house_row = result[(result["bioguide_id"] == "A000001") & (result["chamber"] == "House")]
        assert len(house_row) == 1
        assert house_row.iloc[0]["les_score"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 7–8. Integration tests — require downloaded data files
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_parquet_has_correct_shape() -> None:
    """The saved parquet must have 509 rows (one per registry candidate)."""
    stats = pd.read_parquet(PARQUET_FILE)
    assert len(stats) == 509, f"Expected 509 rows, got {len(stats)}"


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_parquet_has_required_columns() -> None:
    """The saved parquet must contain all required output columns."""
    stats = pd.read_parquet(PARQUET_FILE)
    required = {
        "bioguide_id",
        "career_nominate_dim1",
        "career_nokken_poole_dim1",
        "career_les",
        "career_les2",
        "congresses_served_vv",
        "congresses_served_les",
        "has_legislative_record",
    }
    assert required.issubset(set(stats.columns))


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_parquet_at_least_100_candidates_matched() -> None:
    """At least 100 of 509 candidates should have a VoteView legislative record."""
    stats = pd.read_parquet(PARQUET_FILE)
    n_matched = (stats["congresses_served_vv"] > 0).sum()
    assert n_matched >= 100, f"Expected ≥100 VoteView matches, got {n_matched}"


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_cornyn_is_conservative() -> None:
    """Cornyn's DW-NOMINATE dim1 should be positive (conservative)."""
    stats = pd.read_parquet(PARQUET_FILE)
    assert "C001056" in stats.index, "Cornyn not found in legislative stats"
    assert stats.loc["C001056", "career_nominate_dim1"] > 0.3


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_ossoff_is_liberal() -> None:
    """Ossoff's DW-NOMINATE dim1 should be negative (liberal)."""
    stats = pd.read_parquet(PARQUET_FILE)
    assert "O000174" in stats.index, "Ossoff not found in legislative stats"
    assert stats.loc["O000174", "career_nominate_dim1"] < -0.2


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_murray_has_many_congresses() -> None:
    """Patty Murray (long-serving Senator) should have 10+ Congress rows."""
    stats = pd.read_parquet(PARQUET_FILE)
    assert "M001111" in stats.index
    assert stats.loc["M001111", "congresses_served_vv"] >= 10


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_nominate_dim1_range_plausible() -> None:
    """All non-null DW-NOMINATE dim1 values should be in [-1.5, 1.5]."""
    stats = pd.read_parquet(PARQUET_FILE)
    scored = stats["career_nominate_dim1"].dropna()
    assert (scored >= -1.5).all() and (scored <= 1.5).all(), "DW-NOMINATE dim1 outside plausible range"


@pytest.mark.skipif(not PARQUET_FILE.exists(), reason="legislative_stats.parquet not built yet")
def test_les_scores_are_non_negative() -> None:
    """LES 1.0 and LES 2.0 scores should be non-negative for all matched candidates."""
    stats = pd.read_parquet(PARQUET_FILE)
    les = stats["career_les"].dropna()
    les2 = stats["career_les2"].dropna()
    assert (les >= 0).all(), "LES 1.0 should be non-negative"
    assert (les2 >= 0).all(), "LES 2.0 should be non-negative"
