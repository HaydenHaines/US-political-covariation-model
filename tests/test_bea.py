"""Tests for src/assembly/fetch_bea.py.

All tests use synthetic DataFrames or mock network/file I/O.
No network access or actual BEA files are required.

Coverage:
  1. _clean_fips — FIPS normalization (quoting, padding, length)
  2. _strip_bea_trailer — BEA footer removal
  3. _parse_bea_csv — CSV parsing, suppression codes, year columns
  4. _pick_year — year selection with primary/fallback logic
  5. _compute_growth — growth rate arithmetic, NaN propagation
  6. build_pci_features — output schema, values, growth, dedup
  7. build_gdp_features — GDP per capita, growth, population merge
  8. fetch_bea_features — integration of PCI + GDP, outer join, schema
  9. download_zip — cache hit / miss behavior
  10. read_zip_state_files — file selection from ZIP, error tolerance
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.assembly.fetch_bea import (
    ASSEMBLED_DIR,
    CAINC1_LINE_PCI,
    CAINC1_LINE_POPULATION,
    CAGDP1_LINE_CURRENT_GDP,
    FALLBACK_YEARS,
    GROWTH_LOOKBACK,
    PRIMARY_YEAR,
    RAW_DIR,
    _clean_fips,
    _compute_growth,
    _parse_bea_csv,
    _pick_year,
    _strip_bea_trailer,
    build_gdp_features,
    build_pci_features,
    fetch_bea_features,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_wide_csv(
    fips_list: list[str],
    line_code: int,
    year_values: dict[int, list[float | None]],
    extra_cols: dict | None = None,
) -> bytes:
    """Build a minimal BEA-style wide CSV as bytes.

    Args:
        fips_list: List of 5-digit FIPS codes (already clean).
        line_code: BEA LineCode value.
        year_values: {year: [value_per_fips, ...]} mapping.
        extra_cols: Optional extra metadata columns ({name: [values, ...]}).
    """
    n = len(fips_list)
    rows: dict[str, list] = {
        "GeoFIPS": [f' "{f}"' for f in fips_list],
        "GeoName": [f"County {f}" for f in fips_list],
        "Region": ["5"] * n,
        "TableName": ["TEST"] * n,
        "LineCode": [str(line_code)] * n,
        "IndustryClassification": ["..."] * n,
        "Description": ["Test"] * n,
        "Unit": ["Dollars"] * n,
    }
    if extra_cols:
        rows.update(extra_cols)
    for year, vals in year_values.items():
        rows[str(year)] = [str(v) if v is not None else "(NA)" for v in vals]

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("latin-1")


def _make_cainc1_pci(
    fips_list: list[str],
    year_values: dict[int, list[float | None]],
) -> pd.DataFrame:
    """Make a parsed CAINC1-style DataFrame for LineCode=3 (PCI)."""
    n = len(fips_list)
    data = {
        "county_fips": fips_list,
        "GeoName": [f"County {f}" for f in fips_list],
        "LineCode": [CAINC1_LINE_PCI] * n,
    }
    for year, vals in year_values.items():
        data[str(year)] = [float(v) if v is not None else np.nan for v in vals]
    return pd.DataFrame(data)


def _make_cainc1_pop(
    fips_list: list[str],
    year_values: dict[int, list[float | None]],
) -> pd.DataFrame:
    """Make a parsed CAINC1-style DataFrame for LineCode=2 (population)."""
    n = len(fips_list)
    data = {
        "county_fips": fips_list,
        "GeoName": [f"County {f}" for f in fips_list],
        "LineCode": [CAINC1_LINE_POPULATION] * n,
    }
    for year, vals in year_values.items():
        data[str(year)] = [float(v) if v is not None else np.nan for v in vals]
    return pd.DataFrame(data)


def _make_cagdp1_gdp(
    fips_list: list[str],
    year_values: dict[int, list[float | None]],
) -> pd.DataFrame:
    """Make a parsed CAGDP1-style DataFrame for LineCode=3 (current GDP)."""
    n = len(fips_list)
    data = {
        "county_fips": fips_list,
        "GeoName": [f"County {f}" for f in fips_list],
        "LineCode": [CAGDP1_LINE_CURRENT_GDP] * n,
    }
    for year, vals in year_values.items():
        data[str(year)] = [float(v) if v is not None else np.nan for v in vals]
    return pd.DataFrame(data)


# ── 1. _clean_fips ─────────────────────────────────────────────────────────────

class TestCleanFips:
    def test_strips_quotes_and_spaces(self):
        s = pd.Series([' "01001"', ' "48113"'])
        result = _clean_fips(s)
        assert result.tolist() == ["01001", "48113"]

    def test_zero_pads_short_fips(self):
        s = pd.Series(["1001"])  # missing leading zero
        result = _clean_fips(s)
        assert result.tolist() == ["01001"]

    def test_already_clean_fips_unchanged(self):
        s = pd.Series(["06075", "12086"])
        result = _clean_fips(s)
        assert result.tolist() == ["06075", "12086"]

    def test_state_aggregate_passthrough(self):
        """_clean_fips does not filter — caller filters XX000."""
        s = pd.Series([' "01000"'])
        result = _clean_fips(s)
        assert result.tolist() == ["01000"]


# ── 2. _strip_bea_trailer ──────────────────────────────────────────────────────

class TestStripBeaTrailer:
    def test_removes_note_lines(self):
        text = 'a,b,c\n1,2,3\n"Note: See the included footnote file."\n'
        result = _strip_bea_trailer(text)
        assert '"Note' not in result
        assert "1,2,3" in result

    def test_removes_dataset_name_line(self):
        text = 'a,b\n1,2\n"CAINC1: County personal income summary"\n'
        result = _strip_bea_trailer(text)
        assert "CAINC1:" not in result

    def test_removes_last_updated_line(self):
        text = 'a,b\n1,2\n"Last updated: February 5, 2026"\n'
        result = _strip_bea_trailer(text)
        assert "Last updated" not in result

    def test_removes_attribution_line(self):
        text = 'a,b\n1,2\n"U.S. Bureau of Economic Analysis"\n'
        result = _strip_bea_trailer(text)
        assert "U.S. Bureau" not in result

    def test_preserves_data_rows(self):
        text = 'GeoFIPS,LineCode\n"01001",3\n"Note: footer"\n'
        result = _strip_bea_trailer(text)
        assert '"01001",3' in result

    def test_empty_lines_stripped(self):
        text = 'a,b\n\n1,2\n\n'
        result = _strip_bea_trailer(text)
        assert "\n\n" not in result


# ── 3. _parse_bea_csv ──────────────────────────────────────────────────────────

class TestParseBEACsv:
    def test_state_aggregates_excluded(self):
        raw = _make_wide_csv(
            ["01000", "01001", "01003"],
            line_code=3,
            year_values={2024: [50000.0, 48000.0, 52000.0]},
        )
        df = _parse_bea_csv(raw)
        assert "01000" not in df["county_fips"].values
        assert "01001" in df["county_fips"].values

    def test_year_columns_are_float(self):
        raw = _make_wide_csv(
            ["01001"],
            line_code=3,
            year_values={2024: [55000.0]},
        )
        df = _parse_bea_csv(raw)
        assert df["2024"].dtype.kind == "f"

    def test_suppression_codes_become_nan(self):
        # Build CSV with (NA) in a year column manually
        text = (
            "GeoFIPS,GeoName,Region,TableName,LineCode,IndustryClassification,Description,Unit,2024\n"
            ' "01001",Autauga,5,TEST,3,...,Test,Dollars,(NA)\n'
        )
        df = _parse_bea_csv(text.encode("latin-1"))
        assert pd.isna(df.loc[df["county_fips"] == "01001", "2024"].iloc[0])

    def test_comma_formatted_numbers_parsed(self):
        text = (
            "GeoFIPS,GeoName,Region,TableName,LineCode,IndustryClassification,Description,Unit,2024\n"
            ' "01001",Autauga,5,TEST,3,...,Test,Dollars,"1,234,567"\n'
        )
        df = _parse_bea_csv(text.encode("latin-1"))
        assert df.loc[df["county_fips"] == "01001", "2024"].iloc[0] == pytest.approx(1234567.0)

    def test_fips_zero_padded(self):
        text = (
            "GeoFIPS,GeoName,Region,TableName,LineCode,IndustryClassification,Description,Unit,2024\n"
            ' "1001",Autauga,5,TEST,3,...,Test,Dollars,50000\n'
        )
        df = _parse_bea_csv(text.encode("latin-1"))
        assert df["county_fips"].iloc[0] == "01001"

    def test_line_code_is_numeric(self):
        raw = _make_wide_csv(["01001"], line_code=3, year_values={2024: [50000.0]})
        df = _parse_bea_csv(raw)
        assert df["LineCode"].dtype.kind in ("i", "f", "u")


# ── 4. _pick_year ──────────────────────────────────────────────────────────────

class TestPickYear:
    def _make_df(self, year_cols: list[int], values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({str(y): [v] for y, v in zip(year_cols, values)})

    def test_picks_primary_year_when_available(self):
        df = self._make_df([2022, 2023, 2024], [1.0, 2.0, 3.0])
        assert _pick_year(df, 2024, (2023, 2022)) == 2024

    def test_falls_back_to_first_fallback(self):
        df = self._make_df([2022, 2023], [1.0, 2.0])
        # 2024 not in df
        assert _pick_year(df, 2024, (2023, 2022)) == 2023

    def test_falls_back_to_second_fallback(self):
        df = self._make_df([2022], [1.0])
        assert _pick_year(df, 2024, (2023, 2022)) == 2022

    def test_raises_when_no_year_available(self):
        df = self._make_df([2019], [1.0])
        with pytest.raises(ValueError, match="None of the candidate years"):
            _pick_year(df, 2024, (2023, 2022))

    def test_skips_all_nan_year(self):
        df = pd.DataFrame({"2024": [np.nan], "2023": [45000.0]})
        # 2024 column exists but is all NaN — should fall back to 2023
        assert _pick_year(df, 2024, (2023,)) == 2023


# ── 5. _compute_growth ─────────────────────────────────────────────────────────

class TestComputeGrowth:
    def test_positive_growth(self):
        t = pd.Series([110.0])
        t3 = pd.Series([100.0])
        result = _compute_growth(t, t3)
        assert result.iloc[0] == pytest.approx(0.10)

    def test_negative_growth(self):
        t = pd.Series([90.0])
        t3 = pd.Series([100.0])
        result = _compute_growth(t, t3)
        assert result.iloc[0] == pytest.approx(-0.10)

    def test_zero_base_returns_nan(self):
        t = pd.Series([50.0])
        t3 = pd.Series([0.0])
        result = _compute_growth(t, t3)
        assert pd.isna(result.iloc[0])

    def test_nan_t_returns_nan(self):
        t = pd.Series([np.nan])
        t3 = pd.Series([100.0])
        result = _compute_growth(t, t3)
        assert pd.isna(result.iloc[0])

    def test_nan_t3_returns_nan(self):
        t = pd.Series([110.0])
        t3 = pd.Series([np.nan])
        result = _compute_growth(t, t3)
        assert pd.isna(result.iloc[0])

    def test_negative_base_uses_absolute_value(self):
        """Growth is relative to |base|, not signed base."""
        t = pd.Series([-80.0])
        t3 = pd.Series([-100.0])
        result = _compute_growth(t, t3)
        # (-80 - -100) / |-100| = 20/100 = 0.20
        assert result.iloc[0] == pytest.approx(0.20)

    def test_vectorized_mixed_valid_and_nan(self):
        t = pd.Series([110.0, np.nan, 90.0])
        t3 = pd.Series([100.0, 100.0, 100.0])
        result = _compute_growth(t, t3)
        assert result.iloc[0] == pytest.approx(0.10)
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(-0.10)


# ── 6. build_pci_features ──────────────────────────────────────────────────────

class TestBuildPciFeatures:
    def _make_df(
        self,
        fips: list[str],
        pci_now: list[float | None],
        pci_prior: list[float | None],
    ) -> pd.DataFrame:
        year_t = PRIMARY_YEAR
        year_p = year_t - GROWTH_LOOKBACK
        return _make_cainc1_pci(fips, {year_t: pci_now, year_p: pci_prior})

    def test_output_columns(self):
        df = self._make_df(["01001"], [50000.0], [45000.0])
        result = build_pci_features(df)
        assert set(result.columns) == {"county_fips", "pci", "pci_growth"}

    def test_pci_value_correct(self):
        df = self._make_df(["01001"], [52000.0], [48000.0])
        result = build_pci_features(df)
        assert result.loc[0, "pci"] == pytest.approx(52000.0)

    def test_pci_growth_correct(self):
        # 52000 vs 48000 → growth = (52000 - 48000) / 48000 ≈ 0.0833
        df = self._make_df(["01001"], [52000.0], [48000.0])
        result = build_pci_features(df)
        assert result.loc[0, "pci_growth"] == pytest.approx(4000.0 / 48000.0)

    def test_no_duplicate_county_fips(self):
        fips = ["01001", "01001"]  # duplicate input
        pci_now = [50000.0, 50000.0]
        pci_prior = [45000.0, 45000.0]
        df = _make_cainc1_pci(fips, {PRIMARY_YEAR: pci_now, PRIMARY_YEAR - GROWTH_LOOKBACK: pci_prior})
        result = build_pci_features(df)
        assert result["county_fips"].nunique() == len(result)

    def test_fips_zero_padded(self):
        df = self._make_df(["01001", "06075"], [50000.0, 80000.0], [45000.0, 72000.0])
        result = build_pci_features(df)
        assert (result["county_fips"].str.len() == 5).all()

    def test_nan_pci_propagated(self):
        # Provide two counties: one with valid PCI (so _pick_year succeeds),
        # and one with NaN PCI. The NaN county should appear with NaN pci/pci_growth.
        df = self._make_df(["01001", "01003"], [None, 50000.0], [45000.0, 45000.0])
        result = build_pci_features(df)
        row = result[result["county_fips"] == "01001"].iloc[0]
        assert pd.isna(row["pci"])
        assert pd.isna(row["pci_growth"])

    def test_multiple_counties_independent(self):
        df = self._make_df(
            ["01001", "01003"],
            [50000.0, 60000.0],
            [40000.0, 50000.0],
        )
        result = build_pci_features(df)
        r_01001 = result[result["county_fips"] == "01001"].iloc[0]
        r_01003 = result[result["county_fips"] == "01003"].iloc[0]
        assert r_01001["pci_growth"] == pytest.approx(10000.0 / 40000.0)
        assert r_01003["pci_growth"] == pytest.approx(10000.0 / 50000.0)


# ── 7. build_gdp_features ──────────────────────────────────────────────────────

class TestBuildGdpFeatures:
    def _make_combined(
        self,
        fips: list[str],
        gdp_now: list[float | None],
        gdp_prior: list[float | None],
        pop_now: list[float | None],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        year_t = PRIMARY_YEAR
        year_p = year_t - GROWTH_LOOKBACK
        cagdp1 = _make_cagdp1_gdp(fips, {year_t: gdp_now, year_p: gdp_prior})
        cainc1_pop = _make_cainc1_pop(fips, {year_t: pop_now})
        return cagdp1, cainc1_pop

    def test_output_columns(self):
        cagdp1, cainc1_pop = self._make_combined(
            ["01001"], [1000000.0], [900000.0], [50000.0]
        )
        result = build_gdp_features(cagdp1, cainc1_pop)
        assert set(result.columns) == {"county_fips", "gdp_per_capita", "gdp_growth"}

    def test_gdp_per_capita_correct(self):
        # GDP = 1,000,000 (thousands) → $1B current; pop = 50,000
        # gdp_per_capita = 1,000,000 / 50,000 = 20.0 (thousands/person)
        cagdp1, cainc1_pop = self._make_combined(
            ["01001"], [1_000_000.0], [900_000.0], [50_000.0]
        )
        result = build_gdp_features(cagdp1, cainc1_pop)
        assert result.loc[0, "gdp_per_capita"] == pytest.approx(20.0)

    def test_gdp_growth_correct(self):
        # GDP: 1,100,000 → 1,000,000 (3 years prior) → growth = 10%
        cagdp1, cainc1_pop = self._make_combined(
            ["01001"], [1_100_000.0], [1_000_000.0], [50_000.0]
        )
        result = build_gdp_features(cagdp1, cainc1_pop)
        assert result.loc[0, "gdp_growth"] == pytest.approx(0.10)

    def test_zero_population_produces_nan_gdp_per_capita(self):
        cagdp1, cainc1_pop = self._make_combined(
            ["01001"], [1_000_000.0], [900_000.0], [0.0]
        )
        result = build_gdp_features(cagdp1, cainc1_pop)
        assert pd.isna(result.loc[0, "gdp_per_capita"])

    def test_missing_population_produces_nan_gdp_per_capita(self):
        cagdp1, cainc1_pop = self._make_combined(
            ["01001"], [1_000_000.0], [900_000.0], [None]
        )
        result = build_gdp_features(cagdp1, cainc1_pop)
        assert pd.isna(result.loc[0, "gdp_per_capita"])

    def test_no_duplicate_county_fips(self):
        fips = ["01001", "01001"]
        cagdp1 = _make_cagdp1_gdp(
            fips,
            {PRIMARY_YEAR: [1_000_000.0, 1_000_000.0], PRIMARY_YEAR - GROWTH_LOOKBACK: [900_000.0, 900_000.0]},
        )
        cainc1_pop = _make_cainc1_pop(fips, {PRIMARY_YEAR: [50_000.0, 50_000.0]})
        result = build_gdp_features(cagdp1, cainc1_pop)
        assert result["county_fips"].nunique() == len(result)


# ── 8. fetch_bea_features ──────────────────────────────────────────────────────

class TestFetchBeaFeatures:
    def _synthetic_cainc1(self) -> pd.DataFrame:
        fips = ["01001", "01003", "06075"]
        year_t = PRIMARY_YEAR
        year_p = year_t - GROWTH_LOOKBACK
        pci_rows = _make_cainc1_pci(fips, {year_t: [50000.0, 60000.0, 80000.0], year_p: [45000.0, 54000.0, 72000.0]})
        pop_rows = _make_cainc1_pop(fips, {year_t: [50000.0, 100000.0, 800000.0]})
        return pd.concat([pci_rows, pop_rows], ignore_index=True)

    def _synthetic_cagdp1(self) -> pd.DataFrame:
        fips = ["01001", "01003", "06075"]
        year_t = PRIMARY_YEAR
        year_p = year_t - GROWTH_LOOKBACK
        return _make_cagdp1_gdp(
            fips,
            {year_t: [1_000_000.0, 2_000_000.0, 50_000_000.0], year_p: [900_000.0, 1_800_000.0, 45_000_000.0]},
        )

    @patch("src.assembly.fetch_bea.read_zip_state_files")
    @patch("src.assembly.fetch_bea.download_zip")
    def test_output_columns(self, mock_download, mock_read):
        mock_download.return_value = Path("/fake/path.zip")

        def side_effect(path, prefix):
            if "CAINC1" in prefix:
                return self._synthetic_cainc1()
            return self._synthetic_cagdp1()

        mock_read.side_effect = side_effect
        result = fetch_bea_features()
        assert set(result.columns) == {"county_fips", "pci", "pci_growth", "gdp_per_capita", "gdp_growth"}

    @patch("src.assembly.fetch_bea.read_zip_state_files")
    @patch("src.assembly.fetch_bea.download_zip")
    def test_no_duplicate_county_fips(self, mock_download, mock_read):
        mock_download.return_value = Path("/fake/path.zip")

        def side_effect(path, prefix):
            if "CAINC1" in prefix:
                return self._synthetic_cainc1()
            return self._synthetic_cagdp1()

        mock_read.side_effect = side_effect
        result = fetch_bea_features()
        assert result["county_fips"].nunique() == len(result)

    @patch("src.assembly.fetch_bea.read_zip_state_files")
    @patch("src.assembly.fetch_bea.download_zip")
    def test_county_fips_five_digits(self, mock_download, mock_read):
        mock_download.return_value = Path("/fake/path.zip")

        def side_effect(path, prefix):
            if "CAINC1" in prefix:
                return self._synthetic_cainc1()
            return self._synthetic_cagdp1()

        mock_read.side_effect = side_effect
        result = fetch_bea_features()
        assert (result["county_fips"].str.len() == 5).all()

    @patch("src.assembly.fetch_bea.read_zip_state_files")
    @patch("src.assembly.fetch_bea.download_zip")
    def test_float_columns(self, mock_download, mock_read):
        mock_download.return_value = Path("/fake/path.zip")

        def side_effect(path, prefix):
            if "CAINC1" in prefix:
                return self._synthetic_cainc1()
            return self._synthetic_cagdp1()

        mock_read.side_effect = side_effect
        result = fetch_bea_features()
        for col in ["pci", "pci_growth", "gdp_per_capita", "gdp_growth"]:
            assert result[col].dtype.kind == "f", f"{col} is not float"

    @patch("src.assembly.fetch_bea.read_zip_state_files")
    @patch("src.assembly.fetch_bea.download_zip")
    def test_sorted_by_fips(self, mock_download, mock_read):
        mock_download.return_value = Path("/fake/path.zip")

        def side_effect(path, prefix):
            if "CAINC1" in prefix:
                return self._synthetic_cainc1()
            return self._synthetic_cagdp1()

        mock_read.side_effect = side_effect
        result = fetch_bea_features()
        assert result["county_fips"].tolist() == sorted(result["county_fips"].tolist())


# ── 9. download_zip ────────────────────────────────────────────────────────────

class TestDownloadZip:
    def test_cache_hit_skips_download(self, tmp_path):
        from src.assembly.fetch_bea import download_zip

        cached = tmp_path / "CAINC1.zip"
        cached.write_bytes(b"fake zip content")

        with patch("requests.get") as mock_get:
            result = download_zip("https://example.com/CAINC1.zip", cached, force_refresh=False)

        mock_get.assert_not_called()
        assert result == cached

    def test_cache_miss_calls_requests(self, tmp_path):
        from src.assembly.fetch_bea import download_zip

        dest = tmp_path / "CAINC1.zip"
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake zip content"]
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            result = download_zip("https://example.com/CAINC1.zip", dest, force_refresh=False)

        assert dest.exists()
        assert result == dest

    def test_force_refresh_calls_requests_even_with_cache(self, tmp_path):
        from src.assembly.fetch_bea import download_zip

        cached = tmp_path / "CAINC1.zip"
        cached.write_bytes(b"old content")

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"new content"]
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response) as mock_get:
            download_zip("https://example.com/CAINC1.zip", cached, force_refresh=True)

        mock_get.assert_called_once()

    def test_creates_parent_directory(self, tmp_path):
        from src.assembly.fetch_bea import download_zip

        dest = tmp_path / "new_subdir" / "CAINC1.zip"

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake content"]
        mock_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_response):
            download_zip("https://example.com/CAINC1.zip", dest, force_refresh=False)

        assert dest.parent.exists()


# ── 10. read_zip_state_files ───────────────────────────────────────────────────

class TestReadZipStateFiles:
    def _make_zip(self, file_entries: dict[str, bytes]) -> bytes:
        """Build an in-memory ZIP from a {filename: bytes} dict."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in file_entries.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_reads_state_files_and_skips_all_areas(self, tmp_path):
        from src.assembly.fetch_bea import read_zip_state_files

        al_csv = _make_wide_csv(["01001"], 3, {2024: [50000.0]})
        all_areas_csv = _make_wide_csv(["00000", "01001"], 3, {2024: [None, 50000.0]})

        zip_bytes = self._make_zip({
            "CAINC1_AL_1969_2024.csv": al_csv,
            "CAINC1__ALL_AREAS_1969_2024.csv": all_areas_csv,
            "CAINC1__definition.xml": b"<xml/>",
        })

        zip_path = tmp_path / "CAINC1.zip"
        zip_path.write_bytes(zip_bytes)

        df = read_zip_state_files(zip_path, "CAINC1")

        # ALL_AREAS file (with __) is excluded; county 01001 should appear from AL file
        assert "01001" in df["county_fips"].values

    def test_skips_definition_and_footnotes_files(self, tmp_path):
        from src.assembly.fetch_bea import read_zip_state_files

        al_csv = _make_wide_csv(["01001"], 3, {2024: [50000.0]})
        zip_bytes = self._make_zip({
            "CAINC1_AL_1969_2024.csv": al_csv,
            "CAINC1__definition.xml": b"<xml/>",
            "CAINC1__Footnotes.html": b"<html/>",
        })
        zip_path = tmp_path / "CAINC1.zip"
        zip_path.write_bytes(zip_bytes)

        # Should not crash despite non-CSV files being present
        df = read_zip_state_files(zip_path, "CAINC1")
        assert len(df) > 0

    def test_multiple_state_files_concatenated(self, tmp_path):
        from src.assembly.fetch_bea import read_zip_state_files

        al_csv = _make_wide_csv(["01001", "01003"], 3, {2024: [50000.0, 60000.0]})
        fl_csv = _make_wide_csv(["12001", "12003"], 3, {2024: [55000.0, 65000.0]})

        zip_bytes = self._make_zip({
            "CAINC1_AL_1969_2024.csv": al_csv,
            "CAINC1_FL_1969_2024.csv": fl_csv,
        })
        zip_path = tmp_path / "CAINC1.zip"
        zip_path.write_bytes(zip_bytes)

        df = read_zip_state_files(zip_path, "CAINC1")
        fips_set = set(df["county_fips"].values)
        assert {"01001", "01003", "12001", "12003"}.issubset(fips_set)

    def test_raises_when_no_state_files(self, tmp_path):
        from src.assembly.fetch_bea import read_zip_state_files

        zip_bytes = self._make_zip({"CAINC1__definition.xml": b"<xml/>"})
        zip_path = tmp_path / "CAINC1.zip"
        zip_path.write_bytes(zip_bytes)

        with pytest.raises(ValueError, match="No state CSV files parsed"):
            read_zip_state_files(zip_path, "CAINC1")


# ── 11. Path constants ─────────────────────────────────────────────────────────

class TestPathConstants:
    def test_raw_dir_ends_in_data_raw_bea(self):
        assert RAW_DIR.parts[-3:] == ("data", "raw", "bea")

    def test_assembled_dir_ends_in_data_assembled(self):
        assert ASSEMBLED_DIR.parts[-2:] == ("data", "assembled")

    def test_primary_year_is_recent(self):
        assert PRIMARY_YEAR >= 2023

    def test_growth_lookback_is_three(self):
        assert GROWTH_LOOKBACK == 3

    def test_line_codes_correct(self):
        assert CAINC1_LINE_POPULATION == 2
        assert CAINC1_LINE_PCI == 3
        assert CAGDP1_LINE_CURRENT_GDP == 3
