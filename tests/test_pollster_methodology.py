"""Tests for the pollster methodology tagging script."""

import csv
import io
import tempfile
from pathlib import Path

import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from tag_pollster_methodology import lookup_methodology, tag_csv, METHODOLOGY_MAP


# ---------------------------------------------------------------------------
# lookup_methodology: known pollsters
# ---------------------------------------------------------------------------

class TestLookupMethodology:
    """Unit tests for the lookup_methodology function."""

    # --- Phone ---
    @pytest.mark.parametrize("pollster", [
        "Quinnipiac University",
        "Marist College",
        "Siena College",
        "Siena College/NYT",
        "Fox News",
        "Beacon Research",
        "Monmouth University",
        "CNN",
        "CNN/SSRS",
        "Mason-Dixon",
        "Mitchell Research",
        "Mitchell Research & Communications",
        "EPIC-MRA",
        "Marquette Law",
        "Suffolk",
        "Boston Globe/Suffolk",
        "RMG Research",
        "Fabrizio Lee & Associates",
        "Franklin & Marshall",
        "Franklin & Marshall College",
        "Franklin &amp; Marshall",
        "Hart Research (D)",
        "GBAO (D)",
        "Normington Petts (D)",
        "Saint Anselm College",
        "St. Anselm",
        "UNH",
        "University of New Hampshire",
        "Univ. of New Hampshire",
        "University of North Florida",
        "University of Houston",
        "University of Houston/YouGov",
        "Univ. of Houston",
        "UT Tyler",
        "Univ. of Texas - Tyler",
        "University of Texas - Tyler",
        "Texas Public Opinion Research",
        "Catawba College",
        "Catawba College/YouGov",
        "Glengariff Group",
        "Pan Atlantic Research",
        "Pan Atlantic",
        "Schoen Cooperman Research (D)",
        "NH Journal",
        "NHJournal/Praecones Analytica",
        "Detroit News",
        "Harvard-Harris",
        "Reuters",
        "NPR",
        "Platform Communications",
        "EMC Research",
        "EMC Research (D)",
    ])
    def test_phone_pollsters(self, pollster):
        assert lookup_methodology(pollster) == "phone", f"Expected phone for {pollster!r}"

    # --- IVR ---
    @pytest.mark.parametrize("pollster", [
        "Public Policy Polling",
        "InsiderAdvantage",
        "Cygnal",
    ])
    def test_ivr_pollsters(self, pollster):
        assert lookup_methodology(pollster) == "IVR", f"Expected IVR for {pollster!r}"

    # --- Online ---
    @pytest.mark.parametrize("pollster", [
        "Morning Consult",
        "YouGov",
        "Economist/YouGov",
        "Bowling Green State University/YouGov",
        "Bowling Green Univ.",   # abbreviated form of the BGU/YouGov poll
        "Change Research",
        "Change Research (D)",
        "Data for Progress",
        "AtlasIntel",
        "co/efficient",
        "co/efficient (R)",
        "Decision Desk HQ",
        "Workbench Strategy (D)",
        "WPA Intelligence",
        "Victory Insights",
        "Quantus Insights",
        "Target Insyght",
        "Targoz Market Research",
        "AIF Center (R)",
        "Big Data Poll",
        "Praecones Analytica",
        "Zenith Research (D)",
        "Impact Research (D)",
    ])
    def test_online_pollsters(self, pollster):
        assert lookup_methodology(pollster) == "online", f"Expected online for {pollster!r}"

    # --- Mixed ---
    @pytest.mark.parametrize("pollster", [
        "Emerson College",
        "SurveyUSA",
        "Trafalgar Group",
        "Rasmussen Reports",
        "Susquehanna",
        "Noble Predictive Insights",
        "TIPP Insights",
        "TIPP Insights (R)",
        "Tyson Group",
        "OnMessage",
        "OnMessage Public Strategies (R)",
        "Rosetta Stone",
        "Rosetta Stone Communications (R)",
        "Bendixen & Amandi International",
        "Nexus Strategies/ Strategic Partners Solutions",
        "Nexus/Strategic Partners Solutions",
        "Plymouth Union Public (R)",
        "Data Targeting (R)",
    ])
    def test_mixed_pollsters(self, pollster):
        assert lookup_methodology(pollster) == "mixed", f"Expected mixed for {pollster!r}"

    # --- Unknown ---
    @pytest.mark.parametrize("pollster", [
        "Some Unknown Pollster",
        "XYZ Research LLC",
        "",
        "123 Analytics",
        "yes. every kid.",   # actually in the CSV with no known methodology
    ])
    def test_unknown_pollsters(self, pollster):
        assert lookup_methodology(pollster) == "unknown", f"Expected unknown for {pollster!r}"

    def test_case_insensitive(self):
        """Matching must be case-insensitive."""
        assert lookup_methodology("QUINNIPIAC UNIVERSITY") == "phone"
        assert lookup_methodology("emerson college") == "mixed"
        assert lookup_methodology("YOUGOV") == "online"

    def test_partial_match(self):
        """Substring match should work for embedded pollster names."""
        assert lookup_methodology("Siena College/NYT Upshot") == "phone"
        assert lookup_methodology("CNN/SSRS Nationwide") == "phone"


# ---------------------------------------------------------------------------
# tag_csv: integration tests on a temp CSV
# ---------------------------------------------------------------------------

SAMPLE_CSV_CONTENT = """\
race,geography,geo_level,dem_share,n_sample,date,pollster,notes
2026 AZ Governor,AZ,state,0.5057,850.0,2025-11-10,Emerson College,D=44% R=43%
2026 PA Senate,PA,state,0.52,600.0,2025-12-01,Quinnipiac University,D=52% R=48%
2026 TX Governor,TX,state,0.35,900.0,2025-11-15,YouGov,D=35% R=65%
2026 GA Senate,GA,state,0.49,500.0,2025-10-20,Trafalgar Group,D=49% R=51%
2026 OH Governor,OH,state,0.44,700.0,2025-09-30,Rasmussen Reports,D=44% R=56%
2026 FL Senate,FL,state,0.40,800.0,2025-08-15,Some Unknown Pollster,D=40% R=60%
"""


@pytest.fixture
def sample_csv(tmp_path):
    """Write sample CSV and return its path."""
    path = tmp_path / "polls_test.csv"
    path.write_text(SAMPLE_CSV_CONTENT, encoding="utf-8")
    return str(path)


class TestTagCsv:
    def test_all_rows_have_methodology(self, sample_csv):
        """After tagging, every row must have a non-empty methodology."""
        summary = tag_csv(sample_csv)
        with open(sample_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert "methodology" in row
            assert row["methodology"] != "", f"Empty methodology for pollster {row['pollster']!r}"

    def test_methodology_column_added(self, sample_csv):
        """The methodology column should be present in the written CSV."""
        tag_csv(sample_csv)
        with open(sample_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "methodology" in (reader.fieldnames or [])

    def test_known_pollsters_get_correct_methodology(self, sample_csv):
        """Spot-check that known pollsters receive the right tag."""
        tag_csv(sample_csv)
        with open(sample_csv, newline="", encoding="utf-8") as f:
            rows = {row["pollster"]: row["methodology"] for row in csv.DictReader(f)}

        assert rows["Emerson College"] == "mixed"
        assert rows["Quinnipiac University"] == "phone"
        assert rows["YouGov"] == "online"
        assert rows["Trafalgar Group"] == "mixed"
        assert rows["Rasmussen Reports"] == "mixed"

    def test_unknown_pollster_gets_unknown(self, sample_csv):
        """Pollsters not in the map should be tagged 'unknown'."""
        tag_csv(sample_csv)
        with open(sample_csv, newline="", encoding="utf-8") as f:
            rows = {row["pollster"]: row["methodology"] for row in csv.DictReader(f)}
        assert rows["Some Unknown Pollster"] == "unknown"

    def test_summary_counts(self, sample_csv):
        """Summary dict should correctly count methodologies."""
        summary = tag_csv(sample_csv)
        counts = summary["methodology_counts"]
        assert counts.get("mixed", 0) == 3      # Emerson, Trafalgar, Rasmussen
        assert counts.get("phone", 0) == 1      # Quinnipiac
        assert counts.get("online", 0) == 1     # YouGov
        assert counts.get("unknown", 0) == 1    # Some Unknown Pollster
        assert summary["total_rows"] == 6

    def test_unknown_pollsters_listed(self, sample_csv):
        """Summary should list the names of unknown pollsters."""
        summary = tag_csv(sample_csv)
        assert "Some Unknown Pollster" in summary["unknown_pollsters"]

    def test_idempotent(self, sample_csv):
        """Running the script twice should produce identical output."""
        tag_csv(sample_csv)
        with open(sample_csv, encoding="utf-8") as f:
            first_pass = f.read()
        tag_csv(sample_csv)
        with open(sample_csv, encoding="utf-8") as f:
            second_pass = f.read()
        assert first_pass == second_pass

    def test_existing_columns_preserved(self, sample_csv):
        """Existing columns must not be removed or reordered."""
        tag_csv(sample_csv)
        with open(sample_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        expected = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes", "methodology"]
        assert fieldnames == expected


# ---------------------------------------------------------------------------
# METHODOLOGY_MAP structure validation
# ---------------------------------------------------------------------------

class TestMethodologyMapStructure:
    def test_all_values_are_valid(self):
        """Every entry in METHODOLOGY_MAP must use an allowed value."""
        valid = {"phone", "online", "IVR", "mixed", "unknown"}
        for pattern, method in METHODOLOGY_MAP:
            assert method in valid, f"Invalid method {method!r} for pattern {pattern!r}"

    def test_no_empty_patterns(self):
        """No pattern in the map should be empty."""
        for pattern, method in METHODOLOGY_MAP:
            assert pattern.strip() != "", "Empty pattern found in METHODOLOGY_MAP"

    def test_map_is_list_of_tuples(self):
        """METHODOLOGY_MAP should be a list of 2-tuples."""
        assert isinstance(METHODOLOGY_MAP, list)
        for entry in METHODOLOGY_MAP:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
