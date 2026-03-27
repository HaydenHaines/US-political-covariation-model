"""Load and validate the race registry for a given cycle."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "races"

VALID_RACE_TYPES = {"senate", "governor", "house"}


@dataclass(frozen=True)
class Race:
    race_id: str       # "2026 FL Senate"
    race_type: str     # "senate" | "governor" | "house"
    state: str         # "FL"
    year: int          # 2026
    district: int | None = None  # Only for house races


def load_races(cycle: int) -> list[Race]:
    """Load all races for a given election cycle from the registry CSV."""
    path = _DATA_DIR / f"races_{cycle}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No race registry found at {path}")
    df = pd.read_csv(path, dtype={"year": int})
    races = []
    for _, row in df.iterrows():
        race_type = row["race_type"].lower()
        if race_type not in VALID_RACE_TYPES:
            raise ValueError(f"Invalid race_type '{race_type}' in {row['race_id']}")
        races.append(Race(
            race_id=row["race_id"],
            race_type=race_type,
            state=row["state"],
            year=int(row["year"]),
            district=int(row["district"]) if "district" in row and pd.notna(row.get("district")) else None,
        ))
    return races


def races_for_state(cycle: int, state: str) -> list[Race]:
    """Return all races in a given state for a cycle."""
    return [r for r in load_races(cycle) if r.state == state]


def race_ids(cycle: int) -> list[str]:
    """Return all race_id strings for a cycle."""
    return [r.race_id for r in load_races(cycle)]
