"""Legislative performance stats: DW-NOMINATE ideology scores + Legislative Effectiveness Scores.

Data pipeline:
  1. download_voteview()  — fetches HSall_members.csv from voteview.com
  2. download_les()       — fetches CEL House and Senate Excel files from thelawmakers.org
  3. load_voteview()      — parse VoteView CSV → bioguide_id × congress records
  4. load_les()           — parse CEL Excel files → ICPSR × congress LES records
  5. build_legislative_stats() — join on bioguide_id, compute career summaries

Matching strategy:
  - VoteView provides bioguide_id for all modern members.
  - LES files provide ICPSR (Poole-Rosenthal ID), which maps 1:1 to VoteView ICPSR.
  - We build an ICPSR → bioguide_id lookup from VoteView, then join LES through it.
  - Candidates in the registry with matching bioguide_id are linked via that id.

Output: data/sabermetrics/legislative_stats.parquet
  - One row per candidate (indexed by person_id from registry)
  - Career-averaged ideology and effectiveness metrics
  - See build_legislative_stats() docstring for full column list
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Download URLs
# ---------------------------------------------------------------------------
VOTEVIEW_MEMBERS_URL = "https://voteview.com/static/data/out/members/HSall_members.csv"
LES_HOUSE_URL = "https://thelawmakers.org/wp-content/uploads/2025/06/CELHouse93to118-REVISED-06.26.2025.xlsx"
LES_SENATE_URL = "https://thelawmakers.org/wp-content/uploads/2025/03/CELSenate93to118.xls"

# ---------------------------------------------------------------------------
# VoteView column names (as they appear in HSall_members.csv)
# ---------------------------------------------------------------------------
VV_COL_CONGRESS = "congress"
VV_COL_CHAMBER = "chamber"
VV_COL_ICPSR = "icpsr"
VV_COL_BIOGUIDE = "bioguide_id"
VV_COL_NOMINATE_DIM1 = "nominate_dim1"
VV_COL_NOKKEN_POOLE_DIM1 = "nokken_poole_dim1"

# ---------------------------------------------------------------------------
# LES column indices (0-based) — House Excel file
# Column names are long prose strings; we reference by position for stability.
# ---------------------------------------------------------------------------
HOUSE_COL_NAME = 1  # "Legislator name, as given in THOMAS"
HOUSE_COL_ICPSR = 2  # "ICPSR number, according to Poole and Rosenthal"
HOUSE_COL_CONGRESS = 3  # "Congress number"
HOUSE_COL_LES1 = 52  # "LES 1.0"
HOUSE_COL_LES2 = 68  # "LES 2.0"

# ---------------------------------------------------------------------------
# LES column indices (0-based) — Senate Excel file
# ---------------------------------------------------------------------------
SENATE_COL_LAST = 0  # "last name"
SENATE_COL_FIRST = 1  # "first name"
SENATE_COL_CONGRESS = 3  # "congress number"
SENATE_COL_ICPSR = 5  # "icpsr number"
SENATE_COL_LES1 = 50  # "LES Classic, not including incorporation..."
SENATE_COL_LES2 = 66  # "LES 2.0"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_voteview(dest_dir: Path | None = None) -> Path:
    """Download VoteView HSall_members.csv to data/raw/voteview/.

    This file contains all Congressional members from the 1st Congress forward,
    with DW-NOMINATE ideology scores and Nokken-Poole per-Congress scores.

    Parameters
    ----------
    dest_dir : Path, optional
        Override destination directory. Defaults to data/raw/voteview/.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    if dest_dir is None:
        dest_dir = PROJECT_ROOT / "data" / "raw" / "voteview"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / "HSall_members.csv"
    if dest_file.exists():
        logger.info("VoteView file already present: %s", dest_file)
        return dest_file

    logger.info("Downloading VoteView members from %s", VOTEVIEW_MEMBERS_URL)
    urllib.request.urlretrieve(VOTEVIEW_MEMBERS_URL, dest_file)
    logger.info("Saved VoteView members: %s (%.1f MB)", dest_file, dest_file.stat().st_size / 1e6)
    return dest_file


def download_les(dest_dir: Path | None = None) -> tuple[Path, Path]:
    """Download CEL House and Senate LES Excel files to data/raw/les/.

    Source: thelawmakers.org (Center for Effective Lawmaking).
    Covers 93rd–118th Congress (1973–2024) for both chambers.

    Parameters
    ----------
    dest_dir : Path, optional
        Override destination directory. Defaults to data/raw/les/.

    Returns
    -------
    tuple[Path, Path]
        (house_path, senate_path) — paths to downloaded files.
    """
    if dest_dir is None:
        dest_dir = PROJECT_ROOT / "data" / "raw" / "les"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    house_file = dest_dir / "CELHouse93to118.xlsx"
    senate_file = dest_dir / "CELSenate93to118.xls"

    if not house_file.exists():
        logger.info("Downloading LES House data from %s", LES_HOUSE_URL)
        urllib.request.urlretrieve(LES_HOUSE_URL, house_file)
        logger.info("Saved LES House: %s (%.1f MB)", house_file, house_file.stat().st_size / 1e6)
    else:
        logger.info("LES House file already present: %s", house_file)

    if not senate_file.exists():
        logger.info("Downloading LES Senate data from %s", LES_SENATE_URL)
        urllib.request.urlretrieve(LES_SENATE_URL, senate_file)
        logger.info("Saved LES Senate: %s (%.1f MB)", senate_file, senate_file.stat().st_size / 1e6)
    else:
        logger.info("LES Senate file already present: %s", senate_file)

    return house_file, senate_file


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_voteview(member_file: Path) -> "pd.DataFrame":
    """Parse VoteView HSall_members.csv into a tidy DataFrame.

    Drops the pre-modern era (< 93rd Congress, before 1973) to align with
    the LES coverage window. Filters to rows with a valid bioguide_id only,
    since older members pre-date the bioguide system and cannot be matched
    to our registry.

    Parameters
    ----------
    member_file : Path
        Path to HSall_members.csv.

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id (str), congress (int), chamber (str),
        nominate_dim1 (float), nokken_poole_dim1 (float).
        One row per member × Congress. Members without a bioguide_id
        are excluded.
    """
    import pandas as pd

    df = pd.read_csv(member_file, low_memory=False)

    # Drop rows without a bioguide_id (pre-modern members we can't match)
    df = df[df[VV_COL_BIOGUIDE].notna()].copy()

    # We only need the LES-overlap era (93rd Congress = 1973 onward)
    # Keeping earlier congresses is harmless but wastes memory; drop them.
    MIN_CONGRESS_LES_ERA = 93
    df = df[df[VV_COL_CONGRESS] >= MIN_CONGRESS_LES_ERA].copy()

    keep_cols = [
        VV_COL_BIOGUIDE,
        VV_COL_CONGRESS,
        VV_COL_CHAMBER,
        VV_COL_ICPSR,
        VV_COL_NOMINATE_DIM1,
        VV_COL_NOKKEN_POOLE_DIM1,
    ]
    result = df[keep_cols].copy()
    result[VV_COL_CONGRESS] = result[VV_COL_CONGRESS].astype(int)
    result[VV_COL_ICPSR] = result[VV_COL_ICPSR].astype(int)

    logger.info(
        "Loaded VoteView: %d member-congress rows, %d unique bioguide_ids",
        len(result),
        result[VV_COL_BIOGUIDE].nunique(),
    )
    return result


def _load_les_house(house_file: Path, icpsr_to_bioguide: "dict[int, str]") -> "pd.DataFrame":
    """Parse the CEL House Excel file into a tidy LES DataFrame.

    Column positions are hardcoded because the file uses long prose column
    names that are fragile to reference by string. Positions are validated
    against expected content at load time.

    Parameters
    ----------
    house_file : Path
        Path to CELHouse93to118.xlsx.
    icpsr_to_bioguide : dict
        Mapping from ICPSR integer → bioguide_id string (built from VoteView).

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, congress, chamber, les_score, les2_score.
    """
    import pandas as pd

    df = pd.read_excel(house_file)

    # Validate column positions match expected content (guard against file changes)
    _validate_les_column(df, HOUSE_COL_ICPSR, "icpsr", "House")
    _validate_les_column(df, HOUSE_COL_CONGRESS, "congress", "House")
    _validate_les_column(df, HOUSE_COL_LES1, "les 1", "House")

    rows = []
    for _, row in df.iterrows():
        icpsr_raw = row.iloc[HOUSE_COL_ICPSR]
        if pd.isna(icpsr_raw):
            continue
        icpsr = int(icpsr_raw)
        bioguide_id = icpsr_to_bioguide.get(icpsr)
        if bioguide_id is None:
            continue

        congress_raw = row.iloc[HOUSE_COL_CONGRESS]
        if pd.isna(congress_raw):
            continue

        les1 = _safe_float(row.iloc[HOUSE_COL_LES1])
        les2 = _safe_float(row.iloc[HOUSE_COL_LES2])

        rows.append(
            {
                "bioguide_id": bioguide_id,
                "congress": int(congress_raw),
                "chamber": "House",
                "les_score": les1,
                "les2_score": les2,
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Loaded LES House: %d rows, %d unique bioguide_ids", len(result), result["bioguide_id"].nunique())
    return result


def _load_les_senate(senate_file: Path, icpsr_to_bioguide: "dict[int, str]") -> "pd.DataFrame":
    """Parse the CEL Senate Excel file into a tidy LES DataFrame.

    The Senate file is an older .xls format with different column layout.
    Column positions are validated at load time.

    Parameters
    ----------
    senate_file : Path
        Path to CELSenate93to118.xls.
    icpsr_to_bioguide : dict
        Mapping from ICPSR integer → bioguide_id string (built from VoteView).

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, congress, chamber, les_score, les2_score.
    """
    import pandas as pd

    df = pd.read_excel(senate_file)

    _validate_les_column(df, SENATE_COL_ICPSR, "icpsr", "Senate")
    _validate_les_column(df, SENATE_COL_CONGRESS, "congress", "Senate")
    _validate_les_column(df, SENATE_COL_LES1, "les", "Senate")

    rows = []
    for _, row in df.iterrows():
        icpsr_raw = row.iloc[SENATE_COL_ICPSR]
        if pd.isna(icpsr_raw):
            continue
        icpsr = int(icpsr_raw)
        bioguide_id = icpsr_to_bioguide.get(icpsr)
        if bioguide_id is None:
            continue

        congress_raw = row.iloc[SENATE_COL_CONGRESS]
        if pd.isna(congress_raw):
            continue

        les1 = _safe_float(row.iloc[SENATE_COL_LES1])
        les2 = _safe_float(row.iloc[SENATE_COL_LES2])

        rows.append(
            {
                "bioguide_id": bioguide_id,
                "congress": int(congress_raw),
                "chamber": "Senate",
                "les_score": les1,
                "les2_score": les2,
            }
        )

    result = pd.DataFrame(rows)
    logger.info("Loaded LES Senate: %d rows, %d unique bioguide_ids", len(result), result["bioguide_id"].nunique())
    return result


def load_les(les_dir: Path, voteview_df: "pd.DataFrame") -> "pd.DataFrame":
    """Load and combine House and Senate LES data from thelawmakers.org Excel files.

    Uses VoteView's ICPSR → bioguide_id mapping as the join key, since LES
    files identify members by ICPSR (not bioguide_id directly).

    Parameters
    ----------
    les_dir : Path
        Directory containing CELHouse93to118.xlsx and CELSenate93to118.xls.
    voteview_df : pd.DataFrame
        Output of load_voteview() — used to build ICPSR → bioguide_id lookup.

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id (str), congress (int), chamber (str),
        les_score (float, LES 1.0), les2_score (float, LES 2.0).
        One row per member × Congress. Members with no bioguide_id match
        are excluded.
    """
    import pandas as pd

    les_dir = Path(les_dir)
    house_file = les_dir / "CELHouse93to118.xlsx"
    senate_file = les_dir / "CELSenate93to118.xls"

    if not house_file.exists():
        raise FileNotFoundError(f"LES House file not found: {house_file}. Run download_les() first.")
    if not senate_file.exists():
        raise FileNotFoundError(f"LES Senate file not found: {senate_file}. Run download_les() first.")

    # Build ICPSR → bioguide_id from VoteView (unique per ICPSR)
    icpsr_to_bioguide: dict[int, str] = (
        voteview_df[[VV_COL_ICPSR, VV_COL_BIOGUIDE]]
        .drop_duplicates(subset=[VV_COL_ICPSR])
        .set_index(VV_COL_ICPSR)[VV_COL_BIOGUIDE]
        .to_dict()
    )

    house_df = _load_les_house(house_file, icpsr_to_bioguide)
    senate_df = _load_les_senate(senate_file, icpsr_to_bioguide)

    combined = pd.concat([house_df, senate_df], ignore_index=True)
    logger.info(
        "LES combined: %d rows, %d unique bioguide_ids",
        len(combined),
        combined["bioguide_id"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Career summary builder
# ---------------------------------------------------------------------------


def build_legislative_stats(
    registry: dict,
    voteview_df: "pd.DataFrame",
    les_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Build per-candidate career legislative statistics.

    Joins VoteView DW-NOMINATE and LES data to the candidate registry by
    bioguide_id. For each candidate with a congressional record, computes
    career means across all Congresses served.

    Parameters
    ----------
    registry : dict
        Candidate registry loaded from candidate_registry.json.
        Expected shape: {"persons": {person_id: {name, bioguide_id, ...}}}
    voteview_df : pd.DataFrame
        Output of load_voteview().
    les_df : pd.DataFrame
        Output of load_les().

    Returns
    -------
    pd.DataFrame
        Index: person_id (str, matching registry keys)
        Columns:
          - bioguide_id (str)
          - career_nominate_dim1 (float): mean DW-NOMINATE dim1 across all congresses.
            Negative = liberal, positive = conservative. Null if no VoteView match.
          - career_nokken_poole_dim1 (float): mean Nokken-Poole dim1. Unlike
            DW-NOMINATE (which is lifetime-constant), Nokken-Poole varies per
            Congress, capturing ideological drift.
          - career_les (float): mean LES 1.0 across all congresses. Measures
            legislative effectiveness by bill passage rate (committee → floor → law).
            Null if no LES match.
          - career_les2 (float): mean LES 2.0. Adds credit for cosponsorship
            text incorporated into other members' bills.
          - congresses_served_vv (int): number of Congress-level rows in VoteView.
          - congresses_served_les (int): number of Congress-level rows in LES.
          - has_legislative_record (bool): True if matched in either VoteView
            or LES.
    """
    import pandas as pd

    persons = registry.get("persons", registry)

    # Build VoteView career summaries (mean across congresses)
    vv_career = (
        voteview_df.groupby(VV_COL_BIOGUIDE)
        .agg(
            career_nominate_dim1=(VV_COL_NOMINATE_DIM1, "mean"),
            career_nokken_poole_dim1=(VV_COL_NOKKEN_POOLE_DIM1, "mean"),
            congresses_served_vv=(VV_COL_CONGRESS, "count"),
        )
        .reset_index()
        .rename(columns={VV_COL_BIOGUIDE: "bioguide_id"})
    )

    # Build LES career summaries (mean across congresses, excluding NaN scores)
    les_career = (
        les_df.groupby("bioguide_id")
        .agg(
            career_les=("les_score", "mean"),
            career_les2=("les2_score", "mean"),
            congresses_served_les=("congress", "count"),
        )
        .reset_index()
    )

    rows = []
    for person_id, person in persons.items():
        bioguide_id = person.get("bioguide_id") or person_id

        # Look up VoteView record
        vv_row = vv_career[vv_career["bioguide_id"] == bioguide_id]
        if len(vv_row) > 0:
            vv_data = vv_row.iloc[0]
            career_nominate = vv_data["career_nominate_dim1"]
            career_nokken = vv_data["career_nokken_poole_dim1"]
            congresses_vv = int(vv_data["congresses_served_vv"])
        else:
            career_nominate = float("nan")
            career_nokken = float("nan")
            congresses_vv = 0

        # Look up LES record
        les_row = les_career[les_career["bioguide_id"] == bioguide_id]
        if len(les_row) > 0:
            les_data = les_row.iloc[0]
            career_les = les_data["career_les"]
            career_les2 = les_data["career_les2"]
            congresses_les = int(les_data["congresses_served_les"])
        else:
            career_les = float("nan")
            career_les2 = float("nan")
            congresses_les = 0

        has_record = congresses_vv > 0 or congresses_les > 0

        rows.append(
            {
                "person_id": person_id,
                "bioguide_id": bioguide_id,
                "career_nominate_dim1": career_nominate,
                "career_nokken_poole_dim1": career_nokken,
                "career_les": career_les,
                "career_les2": career_les2,
                "congresses_served_vv": congresses_vv,
                "congresses_served_les": congresses_les,
                "has_legislative_record": has_record,
            }
        )

    result = pd.DataFrame(rows).set_index("person_id")
    n_vv = (result["congresses_served_vv"] > 0).sum()
    n_les = (result["congresses_served_les"] > 0).sum()
    n_record = result["has_legislative_record"].sum()
    logger.info(
        "Legislative stats: %d candidates total, %d matched VoteView, %d matched LES, %d have any record",
        len(result),
        n_vv,
        n_les,
        n_record,
    )
    return result


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_legislative_pipeline() -> "pd.DataFrame":
    """Download data and build legislative_stats.parquet.

    End-to-end pipeline:
      1. Download VoteView HSall_members.csv (if not cached)
      2. Download LES House + Senate Excel files (if not cached)
      3. Load and parse both sources
      4. Build career summary per candidate
      5. Save to data/sabermetrics/legislative_stats.parquet

    Returns
    -------
    pd.DataFrame
        The legislative_stats DataFrame (also saved to disk).
    """
    import json

    registry_path = PROJECT_ROOT / "data" / "sabermetrics" / "candidate_registry.json"
    voteview_dir = PROJECT_ROOT / "data" / "raw" / "voteview"
    les_dir = PROJECT_ROOT / "data" / "raw" / "les"
    output_path = PROJECT_ROOT / "data" / "sabermetrics" / "legislative_stats.parquet"

    with open(registry_path) as f:
        registry = json.load(f)

    # Step 1: Ensure data is downloaded
    voteview_file = download_voteview(voteview_dir)
    download_les(les_dir)

    # Step 2: Load
    voteview_df = load_voteview(voteview_file)
    les_df = load_les(les_dir, voteview_df)

    # Step 3: Build career stats
    stats = build_legislative_stats(registry, voteview_df, les_df)

    # Step 4: Save
    stats.to_parquet(output_path)
    logger.info("Saved legislative stats: %s", output_path)

    return stats


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return f
    except (TypeError, ValueError):
        import math

        return math.nan


def _validate_les_column(df: "pd.DataFrame", col_idx: int, expected_keyword: str, chamber: str) -> None:
    """Assert that a column at col_idx contains expected_keyword in its name.

    This guards against the LES Excel files being reformatted and silently
    shifting column positions.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded Excel DataFrame.
    col_idx : int
        Zero-based column index to check.
    expected_keyword : str
        Substring expected in the column name (case-insensitive).
    chamber : str
        "House" or "Senate" — used in the error message.

    Raises
    ------
    ValueError
        If the column name does not contain the expected keyword.
    """
    col_name = str(df.columns[col_idx]).lower()
    if expected_keyword.lower() not in col_name:
        raise ValueError(
            f"LES {chamber} column {col_idx} expected to contain '{expected_keyword}' "
            f"but got '{df.columns[col_idx]}'. The file format may have changed."
        )


# ---------------------------------------------------------------------------
# Original stub functions — replaced with lightweight wrappers that delegate
# to the new implementation where there is a natural equivalent, or raise
# NotImplementedError for functionality requiring additional data sources
# (amendment records, cosponsor network, roll-call votes, district opinion history).
# ---------------------------------------------------------------------------


def import_les(les_path: str) -> "pd.DataFrame":
    """Import Volden-Wiseman Legislative Effectiveness Scores.

    Source: thelawmakers.org, 93rd-118th Congress, House and Senate.
    Includes LES 1.0 and LES 2.0 (with text incorporation credit).

    This is a convenience wrapper. For full pipeline use, call load_les()
    directly with a voteview_df for ICPSR → bioguide_id resolution.

    Parameters
    ----------
    les_path : str
        Path to a single LES Excel file (House or Senate).

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, congress, chamber, les_score, les2_score.

    Notes
    -----
    Without a VoteView DataFrame for ICPSR bridging, bioguide_id resolution
    requires the VoteView data to be loaded separately. Call load_les() for
    the full pipeline.
    """
    raise NotImplementedError(
        "import_les() is a legacy stub. Use load_les(les_dir, voteview_df) "
        "for full pipeline use, or run_legislative_pipeline() for end-to-end."
    )


def compute_amendment_success_rate(
    amendments: "pd.DataFrame",
    bill_significance: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Amendment Success Rate (ASR) per legislator per Congress.

    Weighted by bill significance (commemorative=1, substantive=5,
    significant=10), following the LES weighting scheme.

    Parameters
    ----------
    amendments : pd.DataFrame
        Columns: sponsor_bioguide, amendment_id, bill_id,
        status (adopted/rejected/withdrawn).
    bill_significance : pd.DataFrame
        Columns: bill_id, significance (commemorative/substantive/significant).

    Returns
    -------
    pd.DataFrame
        Columns: bioguide_id, congress, asr_raw, asr_weighted,
        amendments_offered, amendments_adopted.

    Notes
    -----
    Requires amendment-level congressional data not currently in the pipeline.
    Deferred to a future data acquisition phase.
    """
    raise NotImplementedError(
        "compute_amendment_success_rate() requires amendment-level data "
        "not yet acquired. See docs/SABERMETRICS_ARCHITECTURE.md for roadmap."
    )


def build_cosponsorship_network(
    cosponsorship_data: "pd.DataFrame",
    congress: int,
) -> tuple:
    """Build co-sponsorship network for a given Congress.

    Nodes = legislators. Edges = co-sponsorship relationships,
    weighted by count.

    Parameters
    ----------
    cosponsorship_data : pd.DataFrame
        Columns: bill_id, sponsor_bioguide, cosponsor_bioguide.
    congress : int
        Congress number.

    Returns
    -------
    tuple
        (graph object, centrality_df) where centrality_df has columns:
        bioguide_id, pagerank, betweenness, eigenvector,
        cross_party_cosponsorship_rate.

    Notes
    -----
    Requires cosponsor-level congressional data not currently in the pipeline.
    Deferred to a future data acquisition phase.
    """
    raise NotImplementedError(
        "build_cosponsorship_network() requires cosponsor-level bill data "
        "not yet acquired. See docs/SABERMETRICS_ARCHITECTURE.md for roadmap."
    )


def compute_party_loyalty(
    member_votes: "pd.DataFrame",
    rollcall_metadata: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Bipartisan Deviation Rate (BDR) and Strategic Defection/Loyalty (SDL).

    BDR = party_defections / total_party_line_votes

    SDL uses Snyder-Groseclose (2000) methodology: partition roll calls
    into close votes (margin < 65-35) and lopsided votes (> 65-35).
    SDL = defection_rate_close / defection_rate_lopsided.
    SDL > 1 = principled dissenter. SDL < 1 = strategic brand-builder.

    Parameters
    ----------
    member_votes : pd.DataFrame
        Per-member per-vote records from VoteView.
        Columns: icpsr_id, rollnumber, cast_code.
    rollcall_metadata : pd.DataFrame
        Roll call metadata from VoteView.
        Columns: rollnumber, congress, chamber, yea_count, nay_count.

    Returns
    -------
    pd.DataFrame
        Columns: icpsr_id, congress, bdr, sdl,
        defections_close, defections_lopsided,
        total_close_votes, total_lopsided_votes.

    Notes
    -----
    Requires VoteView member-vote records (HSall_rollcalls.csv) not yet
    downloaded. Deferred to a future data acquisition phase.
    """
    raise NotImplementedError(
        "compute_party_loyalty() requires VoteView roll-call vote data "
        "(HSall_rollcalls.csv) not yet in the pipeline. "
        "See docs/SABERMETRICS_ARCHITECTURE.md for roadmap."
    )


def compute_responsiveness_index(
    nokken_poole_history: "pd.DataFrame",
    district_opinion_history: "pd.DataFrame",
) -> "pd.DataFrame":
    """Compute Responsiveness Index (RI).

    RI = correlation(delta_Nokken_Poole, delta_district_opinion)
    across Congresses for the same legislator.

    Parameters
    ----------
    nokken_poole_history : pd.DataFrame
        Columns: icpsr_id, congress, nokken_poole_dim1.
    district_opinion_history : pd.DataFrame
        Columns: district_id, congress, median_opinion (from CES MRP).

    Returns
    -------
    pd.DataFrame
        Columns: icpsr_id, ri, n_congresses, delta_nominate_trajectory,
        delta_opinion_trajectory.

    Notes
    -----
    Requires district-level MRP opinion estimates not yet in the pipeline.
    Nokken-Poole history is available via load_voteview(). The missing input
    is district_opinion_history from CES. Deferred to a future phase.
    """
    raise NotImplementedError(
        "compute_responsiveness_index() requires district-level MRP opinion "
        "estimates (CES) not yet in the pipeline. "
        "See docs/SABERMETRICS_ARCHITECTURE.md for roadmap."
    )
