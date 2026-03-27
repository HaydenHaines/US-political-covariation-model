import pytest
from src.assembly.define_races import load_races, Race


def test_load_races_returns_list_of_race():
    races = load_races(2026)
    assert len(races) > 0
    assert all(isinstance(r, Race) for r in races)


def test_race_has_required_fields():
    races = load_races(2026)
    r = races[0]
    assert hasattr(r, "race_id")
    assert hasattr(r, "race_type")
    assert hasattr(r, "state")
    assert hasattr(r, "year")


def test_race_id_format():
    """Race IDs must match the label convention: 'YYYY ST RaceType'."""
    races = load_races(2026)
    for r in races:
        parts = r.race_id.split(" ")
        assert len(parts) == 3, f"Bad race_id format: {r.race_id}"
        assert parts[0] == "2026"
        assert len(parts[1]) == 2 and parts[1].isupper()
        assert parts[2] in ("Senate", "Governor")


def test_no_duplicate_race_ids():
    races = load_races(2026)
    ids = [r.race_id for r in races]
    assert len(ids) == len(set(ids)), f"Duplicate race IDs: {set(x for x in ids if ids.count(x) > 1)}"


def test_senate_count():
    """2026 has 35 Senate races (33 Class II + 2 specials: FL, OH)."""
    races = load_races(2026)
    senate = [r for r in races if r.race_type == "senate"]
    assert 30 <= len(senate) <= 40


def test_governor_count():
    """2026 has 36 Governor races."""
    races = load_races(2026)
    gov = [r for r in races if r.race_type == "governor"]
    assert 30 <= len(gov) <= 40


def test_all_states_represented():
    """Most states should appear at least once."""
    races = load_races(2026)
    states = {r.state for r in races}
    assert len(states) >= 45  # Some states may have no race in 2026
