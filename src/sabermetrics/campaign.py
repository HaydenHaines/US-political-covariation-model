"""Campaign finance statistics for the sabermetrics pipeline.

Computes three campaign finance stats from FEC bulk summary data (weball files):

    SDR  (Small-Dollar Ratio)      = unitemized_receipts / total_individual_contributions
    FER  (Fundraising Efficiency)  = (net_raised - fundraising_costs) / net_raised
    Burn Rate                      = total_disbursements / total_receipts

These are computed per candidate × cycle and written to
data/sabermetrics/campaign_stats.parquet.

Data pipeline:
  1. download_fec_bulk() in ingest.py fetches weball{YY}.zip + cm{YY}.zip
  2. load_weball_file() parses the pipe-delimited FEC summary text file
  3. load_committee_master() parses cm{YY}.txt to map committee → candidate
  4. build_campaign_stats() joins FEC records to the candidate registry via
     bioguide → FEC candidate ID crosswalk (from congress-legislators YAML)
  5. For each matched candidate-cycle, compute SDR, FER, burn_rate

FEC data sources:
  https://www.fec.gov/campaign-finance-data/all-candidates-file-description/
  https://www.fec.gov/campaign-finance-data/committee-master-file-description/
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Column definitions for FEC weball (all-candidates summary) file
# ---------------------------------------------------------------------------
# Source: https://www.fec.gov/campaign-finance-data/all-candidates-file-description/
# The weball file is pipe-delimited with no header row.  Column positions
# are fixed by the FEC spec.  We select only the columns we need.
#
# Selected column indices and names (0-indexed):
#   0  CAND_ID              — FEC candidate ID (e.g., "S4GA00161")
#   1  CAND_NAME
#   2  CAND_ICI             — I=incumbent, C=challenger, O=open seat
#   3  PTY_CD               — party code
#   4  CAND_PTY_AFFILIATION — party abbreviation
#   5  TTL_RECEIPTS         — total receipts
#   6  TRANS_FROM_AUTH      — transfers from authorized committees
#   7  TTL_DISB             — total disbursements
#   8  TRANS_TO_AUTH        — transfers to authorized committees
#   9  COH_BOP              — cash on hand beginning of period
#  10  COH_COP              — cash on hand close of period
#  11  CAND_CONTRIB         — contributions from candidate
#  12  CAND_LOANS           — loans from candidate
#  13  OTHER_LOANS          — other loans
#  14  CAND_LOAN_REPAY      — candidate loan repayments
#  15  OTHER_LOAN_REPAY     — other loan repayments
#  16  DEBTS_OWED_BY        — debts owed by committee
#  17  TTL_INDIV_CONTRIB    — total individual contributions  ← SDR denominator
#  18  CAND_OFFICE_ST       — state (2-letter)
#  19  CAND_OFFICE_DISTRICT — district number
#  20  SPEC_ELECTION        — special election flag
#  21  PRIM_ELECTION        — primary election flag
#  22  RUN_ELECTION         — runoff election flag
#  23  GEN_ELECTION         — general election flag
#  24  GEN_ELECTION_PRECENT — general election win %
#  25  OTHER_POL_CMTE_CONTRIB — contributions from other political committees
#  26  POL_PTY_CONTRIB      — political party contributions
#  27  CVG_END_DT           — coverage end date
#  28  INDIV_REFUNDS        — individual refunds
#  29  CMTE_REFUNDS         — committee refunds
#
# NOTE: The FEC does NOT separately report "unitemized" receipts in weball.
# The closest proxy is: unitemized ≈ TTL_INDIV_CONTRIB - (itemized itemizations).
# However, weball does not include itemized individual totals separately.
#
# WORKAROUND: FEC Form 3/3P Schedule A provides unitemized amounts, but those
# are in the individual contributions file (itcont{YY}.zip, ~3GB).
# For weball-only SDR, we use a proxy:
#   SDR ≈ (TTL_INDIV_CONTRIB - POL_PTY_CONTRIB - OTHER_POL_CMTE_CONTRIB) / TTL_INDIV_CONTRIB
# This is not exact but is computable from weball without downloading ~3GB files.
#
# A better SDR requires the itemized contributions file — documented as a TODO.

_WEBALL_SELECTED_COLS = {
    0: "fec_candidate_id",
    1: "candidate_name",
    2: "incumbent_challenger_status",
    3: "party_code",
    4: "party_affiliation",
    5: "total_receipts",
    6: "transfers_from_auth",
    7: "total_disbursements",
    8: "transfers_to_auth",
    9: "cash_on_hand_bop",
    10: "cash_on_hand_cop",
    11: "candidate_contributions",
    12: "candidate_loans",
    13: "other_loans",
    14: "candidate_loan_repayments",
    15: "other_loan_repayments",
    16: "debts_owed",
    17: "total_individual_contributions",
    18: "state",
    19: "district",
    20: "special_election",
    21: "primary_election",
    22: "runoff_election",
    23: "general_election",
    24: "general_election_pct",
    25: "other_committee_contributions",
    26: "party_committee_contributions",
    27: "coverage_end_date",
    28: "individual_refunds",
    29: "committee_refunds",
}

# Columns that hold dollar amounts (should be coerced to float, not int,
# because FEC sometimes uses decimal notation for cents)
_DOLLAR_COLUMNS = {
    "total_receipts",
    "transfers_from_auth",
    "total_disbursements",
    "transfers_to_auth",
    "cash_on_hand_bop",
    "cash_on_hand_cop",
    "candidate_contributions",
    "candidate_loans",
    "other_loans",
    "candidate_loan_repayments",
    "other_loan_repayments",
    "debts_owed",
    "total_individual_contributions",
    "other_committee_contributions",
    "party_committee_contributions",
    "individual_refunds",
    "committee_refunds",
}

# ---------------------------------------------------------------------------
# Column definitions for FEC committee master (cm{YY}.zip) file
# ---------------------------------------------------------------------------
# Source: https://www.fec.gov/campaign-finance-data/committee-master-file-description/
# Pipe-delimited, no header.  We only need CMTE_ID (col 0) and CAND_ID (col 14).

_CM_SELECTED_COLS = {
    0: "committee_id",
    1: "committee_name",
    10: "committee_type",
    14: "fec_candidate_id",
}

# ---------------------------------------------------------------------------
# FEC candidate office codes used to filter to federal races
# ---------------------------------------------------------------------------
# FEC CAND_ID prefix encodes office: H=House, S=Senate, P=President
# Governors have no FEC candidate ID (they file at the state level).

_FEDERAL_OFFICE_PREFIXES = {"H", "S", "P"}


# ---------------------------------------------------------------------------
# Loader: weball file
# ---------------------------------------------------------------------------


def load_weball_file(zip_path: str | Path) -> "pd.DataFrame":
    """Parse an FEC weball bulk zip into a DataFrame.

    The weball file has no header row and uses pipe (|) as delimiter.
    Columns are fixed by FEC spec — see _WEBALL_SELECTED_COLS above.

    Parameters
    ----------
    zip_path : str | Path
        Path to a weball{YY}.zip file (e.g., weball22.zip).

    Returns
    -------
    pd.DataFrame
        One row per FEC candidate record. Dollar columns are float.
        fec_candidate_id is the primary key.
    """
    import pandas as pd

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"weball zip not found: {zip_path}")

    logger.info("Loading weball file from %s", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        # The text file inside has the same stem as the zip (e.g., weball22.txt)
        txt_name = zip_path.stem + ".txt"
        if txt_name not in zf.namelist():
            # Some FEC zips use all-caps or slightly different names
            available = zf.namelist()
            txt_candidates = [n for n in available if n.lower().endswith(".txt")]
            if not txt_candidates:
                raise ValueError(
                    f"No .txt file found in {zip_path}. Contents: {available}"
                )
            txt_name = txt_candidates[0]
            logger.warning("Expected %s.txt but found %s", zip_path.stem, txt_name)

        with zf.open(txt_name) as f:
            raw_bytes = f.read()

    # FEC files use Latin-1 encoding (some candidate names have accented chars)
    raw_text = raw_bytes.decode("latin-1")

    # Parse pipe-delimited, no header, max 30 columns (spec has 30)
    # Use only the columns we need to keep memory low.
    # Guard against empty files (e.g., cycle with no data).
    usecols = sorted(_WEBALL_SELECTED_COLS.keys())
    if not raw_text.strip():
        logger.warning("weball file in %s is empty — returning empty DataFrame", zip_path.name)
        return pd.DataFrame(columns=list(_WEBALL_SELECTED_COLS.values()))

    df = pd.read_csv(
        io.StringIO(raw_text),
        sep="|",
        header=None,
        usecols=usecols,
        dtype=str,  # Read everything as str first; we'll coerce below
        on_bad_lines="warn",
    )

    # Rename columns from positional integers to descriptive names
    df.rename(columns=_WEBALL_SELECTED_COLS, inplace=True)

    # Coerce dollar columns to float (blank/empty → NaN)
    present_dollar_cols = _DOLLAR_COLUMNS & set(df.columns)
    for col in present_dollar_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace from string ID columns
    df["fec_candidate_id"] = df["fec_candidate_id"].str.strip()
    df["state"] = df["state"].str.strip()

    # Drop rows with no candidate ID (malformed lines)
    df = df[df["fec_candidate_id"].notna() & (df["fec_candidate_id"] != "")]

    logger.info("  Loaded %d candidate records from %s", len(df), zip_path.name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loader: committee master file
# ---------------------------------------------------------------------------


def load_committee_master(zip_path: str | Path) -> "pd.DataFrame":
    """Parse an FEC committee master zip into a DataFrame.

    Used to link a principal campaign committee ID to a candidate ID.
    This is needed when a candidate's authorized committee filed under
    a committee ID that differs from the candidate ID prefix.

    In practice, for Senate/House candidates the weball file already
    carries the candidate ID directly, so this file is used for
    cross-validation and for future extensions (PAC linkage).

    Parameters
    ----------
    zip_path : str | Path
        Path to cm{YY}.zip (e.g., cm22.zip).

    Returns
    -------
    pd.DataFrame
        Columns: committee_id, committee_name, committee_type, fec_candidate_id.
    """
    import pandas as pd

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Committee master zip not found: {zip_path}")

    logger.info("Loading committee master from %s", zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        txt_name = zip_path.stem + ".txt"
        if txt_name not in zf.namelist():
            available = zf.namelist()
            txt_candidates = [n for n in available if n.lower().endswith(".txt")]
            if not txt_candidates:
                raise ValueError(
                    f"No .txt file found in {zip_path}. Contents: {available}"
                )
            txt_name = txt_candidates[0]

        with zf.open(txt_name) as f:
            raw_bytes = f.read()

    raw_text = raw_bytes.decode("latin-1")

    # Committee master has 15+ columns; we only need cols 0, 1, 10, 14
    usecols = sorted(_CM_SELECTED_COLS.keys())
    df = pd.read_csv(
        io.StringIO(raw_text),
        sep="|",
        header=None,
        usecols=usecols,
        dtype=str,
        on_bad_lines="warn",
    )
    df.rename(columns=_CM_SELECTED_COLS, inplace=True)

    # Strip whitespace
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Keep only rows with a linked candidate (many committees have no candidate)
    df = df[df["fec_candidate_id"].notna() & (df["fec_candidate_id"] != "")]

    logger.info("  Loaded %d committee-candidate links from %s", len(df), zip_path.name)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Finance stat computations
# ---------------------------------------------------------------------------


def compute_sdr(fec_summary: "pd.Series") -> float | None:
    """Compute Small-Dollar Ratio from a single FEC candidate summary row.

    SDR = unitemized_receipts / total_individual_contributions

    Interpretation: fraction of individual contributions that came from
    small donors (unitemized — under $200 threshold, not reported individually
    on Schedule A). Higher SDR = more grassroots fundraising base.

    Weball-proxy: since weball does not separately report unitemized amounts,
    we approximate:
        unitemized ≈ total_individual_contributions
                      - other_committee_contributions
                      - party_committee_contributions

    This understates SDR because "other committee contributions" in weball
    are contributions from PACs/party committees, not small donors — but the
    proxy is reasonable given the data available without downloading ~3GB
    individual contribution files.

    Returns None if denominator is zero or missing (avoids divide-by-zero).

    Parameters
    ----------
    fec_summary : pd.Series
        Row from load_weball_file() output.

    Returns
    -------
    float | None
        SDR in [0, 1], or None if not computable.
    """
    total_indiv = _safe_float(fec_summary.get("total_individual_contributions"))
    if total_indiv is None or total_indiv <= 0:
        return None

    # Subtract large-donor / party contributions to approximate small-dollar
    other_cmte = _safe_float(fec_summary.get("other_committee_contributions")) or 0.0
    party_cmte = _safe_float(fec_summary.get("party_committee_contributions")) or 0.0

    # Unitemized proxy = total_indiv minus the identifiable large-donor flows
    # Clamp to [0, total_indiv] — negative values indicate data anomalies
    unitemized_proxy = max(0.0, total_indiv - other_cmte - party_cmte)

    return unitemized_proxy / total_indiv


def compute_fer(fec_summary: "pd.Series") -> float | None:
    """Compute Fundraising Efficiency Ratio from a single FEC candidate summary row.

    FER = (net_raised - fundraising_costs) / net_raised
        = (total_receipts - candidate_contributions - candidate_loans
           - other_loans - fundraising_costs) / net_raised

    Interpretation: fraction of raised money that went to actual campaigning
    (vs. being consumed by fundraising overhead). Higher FER = more efficient.

    Weball does not separate "fundraising costs" from other operating expenses.
    Proxy: net_raised = total_receipts - candidate_self_funding
           fundraising_costs ≈ individual_refunds + committee_refunds
           (refunds are a structural fundraising cost; admin overhead is unknown)
    FER ≈ (net_raised - refunds) / net_raised

    Returns None if net_raised <= 0.

    Parameters
    ----------
    fec_summary : pd.Series
        Row from load_weball_file() output.

    Returns
    -------
    float | None
        FER in (-inf, 1], or None if not computable.
    """
    total_receipts = _safe_float(fec_summary.get("total_receipts"))
    if total_receipts is None or total_receipts <= 0:
        return None

    # Self-funding = candidate contributions + loans (these are not "raised" from donors)
    cand_contrib = _safe_float(fec_summary.get("candidate_contributions")) or 0.0
    cand_loans = _safe_float(fec_summary.get("candidate_loans")) or 0.0
    other_loans = _safe_float(fec_summary.get("other_loans")) or 0.0

    net_raised = total_receipts - cand_contrib - cand_loans - other_loans
    if net_raised <= 0:
        return None

    # Fundraising cost proxy: refunds paid back to donors/committees
    indiv_refunds = _safe_float(fec_summary.get("individual_refunds")) or 0.0
    cmte_refunds = _safe_float(fec_summary.get("committee_refunds")) or 0.0
    fundraising_costs = indiv_refunds + cmte_refunds

    return (net_raised - fundraising_costs) / net_raised


def compute_burn_rate(fec_summary: "pd.Series") -> float | None:
    """Compute campaign burn rate from a single FEC candidate summary row.

    Burn Rate = total_disbursements / total_receipts

    Interpretation: fraction of raised money already spent. >1.0 means the
    campaign is spending more than it's raising (drawing down cash reserves
    or taking on debt). Burn rate is cycle-to-date, not a real-time snapshot.

    Returns None if total_receipts is zero or missing.

    Parameters
    ----------
    fec_summary : pd.Series
        Row from load_weball_file() output.

    Returns
    -------
    float | None
        Burn rate >= 0, or None if not computable.
    """
    total_receipts = _safe_float(fec_summary.get("total_receipts"))
    total_disbursements = _safe_float(fec_summary.get("total_disbursements"))

    if total_receipts is None or total_receipts <= 0:
        return None
    if total_disbursements is None:
        return None

    return total_disbursements / total_receipts


# ---------------------------------------------------------------------------
# Registry → FEC ID crosswalk
# ---------------------------------------------------------------------------


def build_fec_id_crosswalk(
    registry: dict,
    legislators_dir: str | Path = "data/raw/congress-legislators",
) -> dict[str, list[str]]:
    """Build a mapping from registry person_id → list of FEC candidate IDs.

    FEC candidate IDs are stored in the congress-legislators YAML under
    id.fec (a list, since a person can have multiple FEC IDs across cycles).
    We load those IDs and join to the registry by bioguide_id.

    Candidates without a bioguide_id (governors, challengers who never served
    in Congress) have no FEC candidate ID available from this crosswalk and
    will be matched by name + state if possible (future work). They are
    returned with an empty list.

    Parameters
    ----------
    registry : dict
        Loaded candidate_registry.json ({"persons": {...}, "_meta": {...}}).
    legislators_dir : str | Path
        Directory containing legislators-current.yaml and legislators-historical.yaml.

    Returns
    -------
    dict[str, list[str]]
        Maps person_id → list[fec_candidate_id].
        Governors and challengers without bioguide IDs get empty lists.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required: uv add pyyaml") from exc

    legislators_dir = Path(legislators_dir)

    # Build bioguide → fec_ids map from congress-legislators YAML
    bioguide_to_fec: dict[str, list[str]] = {}
    for filename in ["legislators-current.yaml", "legislators-historical.yaml"]:
        path = legislators_dir / filename
        if not path.exists():
            logger.warning("Missing %s — FEC ID crosswalk will be incomplete", path)
            continue
        with open(path) as f:
            records = yaml.safe_load(f)
        for rec in records:
            bioguide = rec.get("id", {}).get("bioguide")
            if not bioguide:
                continue
            fec_ids = rec.get("id", {}).get("fec", [])
            if fec_ids:
                bioguide_to_fec[bioguide] = list(fec_ids)

    # Map registry person_id → FEC IDs via bioguide.
    # If no YAML files were found, bioguide_to_fec is empty — all persons
    # return empty FEC ID lists rather than raising.
    persons = registry.get("persons", {})
    crosswalk: dict[str, list[str]] = {}
    for person_id, person in persons.items():
        bioguide = person.get("bioguide_id")
        if bioguide and bioguide in bioguide_to_fec:
            crosswalk[person_id] = bioguide_to_fec[bioguide]
        else:
            crosswalk[person_id] = []

    n_matched = sum(1 for ids in crosswalk.values() if ids)
    logger.info(
        "FEC ID crosswalk: %d / %d registry persons have FEC IDs",
        n_matched,
        len(crosswalk),
    )
    return crosswalk


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_campaign_stats(
    candidate_registry: dict,
    cycles: list[str],
    fec_data_dir: str | Path = "data/raw/fec",
    legislators_dir: str | Path = "data/raw/congress-legislators",
    output_path: str | Path = "data/sabermetrics/campaign_stats.parquet",
    download_if_missing: bool = True,
) -> "pd.DataFrame":
    """Build campaign finance stats for all registry candidates.

    Full pipeline:
      1. For each cycle, load weball{YY}.zip (download if missing)
      2. Build person_id → FEC candidate IDs crosswalk
      3. For each person × cycle with a matching FEC record, compute
         SDR, FER, burn_rate
      4. State-only candidates (governors, challengers without FEC IDs)
         appear in output with NaN stats and a "no_fec_record" flag
      5. Write output to campaign_stats.parquet

    Parameters
    ----------
    candidate_registry : dict
        Loaded candidate_registry.json.
    cycles : list[str]
        Election cycles to process, e.g. ["2022", "2024"].
    fec_data_dir : str | Path
        Directory containing (or to store) FEC bulk zip files.
    legislators_dir : str | Path
        Directory containing congress-legislators YAML files.
    output_path : str | Path
        Where to write the output parquet file.
    download_if_missing : bool
        Whether to download missing FEC zip files automatically.

    Returns
    -------
    pd.DataFrame
        One row per person × cycle. Columns:
          person_id, name, party, cycle, fec_candidate_id,
          total_receipts, total_disbursements,
          total_individual_contributions,
          sdr, fer, burn_rate,
          has_fec_record (bool)
    """
    import pandas as pd

    fec_data_dir = Path(fec_data_dir)
    output_path = Path(output_path)

    # --- Step 1: Download FEC files if missing ---
    if download_if_missing:
        from src.sabermetrics.ingest import download_fec_bulk

        # Only download weball; cm is optional for this pipeline
        missing_cycles = []
        for cycle in cycles:
            cycle_suffix = cycle[-2:]
            weball_path = fec_data_dir / f"weball{cycle_suffix}.zip"
            if not weball_path.exists():
                missing_cycles.append(cycle)
        if missing_cycles:
            logger.info("Downloading missing FEC weball files for cycles: %s", missing_cycles)
            download_fec_bulk(missing_cycles, output_dir=str(fec_data_dir), file_types=["weball"])

    # --- Step 2: Build FEC ID crosswalk ---
    fec_crosswalk = build_fec_id_crosswalk(candidate_registry, legislators_dir=legislators_dir)

    # --- Step 3: Load weball data for each cycle ---
    # Build a lookup: fec_candidate_id → weball row, per cycle
    weball_by_cycle: dict[str, dict[str, "pd.Series"]] = {}
    for cycle in cycles:
        cycle_suffix = cycle[-2:]
        weball_path = fec_data_dir / f"weball{cycle_suffix}.zip"
        if not weball_path.exists():
            logger.warning("weball%s.zip not found — cycle %s will have no FEC data", cycle_suffix, cycle)
            weball_by_cycle[cycle] = {}
            continue

        weball_df = load_weball_file(weball_path)
        # Index by FEC candidate ID for O(1) lookup
        weball_by_cycle[cycle] = {
            row["fec_candidate_id"]: row
            for _, row in weball_df.iterrows()
        }
        logger.info("  Cycle %s: %d FEC candidate records loaded", cycle, len(weball_by_cycle[cycle]))

    # --- Step 4: Match registry candidates to FEC records and compute stats ---
    persons = candidate_registry.get("persons", {})
    rows: list[dict] = []

    for person_id, person in persons.items():
        fec_ids = fec_crosswalk.get(person_id, [])
        name = person.get("name", "")
        party = person.get("party", "")

        for cycle in cycles:
            cycle_int = int(cycle)
            weball_lookup = weball_by_cycle.get(cycle, {})

            # Find the FEC record for this person in this cycle.
            # A person may have multiple FEC IDs (e.g., ran in different offices).
            # We pick the one with the highest total_receipts for this cycle.
            matched_row: pd.Series | None = None
            matched_fec_id: str | None = None

            for fec_id in fec_ids:
                row = weball_lookup.get(fec_id)
                if row is None:
                    continue
                # Prefer the record with higher total_receipts
                if matched_row is None:
                    matched_row = row
                    matched_fec_id = fec_id
                elif _safe_float(row.get("total_receipts"), 0.0) > _safe_float(
                    matched_row.get("total_receipts"), 0.0
                ):
                    matched_row = row
                    matched_fec_id = fec_id

            # Check if this candidate actually ran in this cycle
            # (some candidates appear in registry for 2022 only but we're processing 2024 too)
            candidate_cycle_years = {r["year"] for r in person.get("races", [])}
            ran_in_cycle = cycle_int in candidate_cycle_years

            has_fec_record = matched_row is not None

            if matched_row is not None:
                sdr = compute_sdr(matched_row)
                fer = compute_fer(matched_row)
                burn_rate = compute_burn_rate(matched_row)
                total_receipts = _safe_float(matched_row.get("total_receipts"))
                total_disbursements = _safe_float(matched_row.get("total_disbursements"))
                total_indiv = _safe_float(matched_row.get("total_individual_contributions"))
            else:
                sdr = None
                fer = None
                burn_rate = None
                total_receipts = None
                total_disbursements = None
                total_indiv = None

            rows.append(
                {
                    "person_id": person_id,
                    "name": name,
                    "party": party,
                    "cycle": cycle_int,
                    "ran_in_cycle": ran_in_cycle,
                    "fec_candidate_id": matched_fec_id,
                    "total_receipts": total_receipts,
                    "total_disbursements": total_disbursements,
                    "total_individual_contributions": total_indiv,
                    "sdr": sdr,
                    "fer": fer,
                    "burn_rate": burn_rate,
                    "has_fec_record": has_fec_record,
                }
            )

    # Define the output schema explicitly so an empty registry still produces
    # a correctly-typed empty DataFrame (rather than a zero-column DataFrame).
    _OUTPUT_DTYPES = {
        "person_id": "object",
        "name": "object",
        "party": "object",
        "cycle": "int64",
        "ran_in_cycle": "bool",
        "fec_candidate_id": "object",
        "total_receipts": "float64",
        "total_disbursements": "float64",
        "total_individual_contributions": "float64",
        "sdr": "float64",
        "fer": "float64",
        "burn_rate": "float64",
        "has_fec_record": "bool",
    }

    if rows:
        df = pd.DataFrame(rows)
        # Coerce numeric columns that may have Python None → NaN
        for col in ("sdr", "fer", "burn_rate", "total_receipts", "total_disbursements",
                    "total_individual_contributions"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df = pd.DataFrame(columns=list(_OUTPUT_DTYPES.keys()))
        for col, dtype in _OUTPUT_DTYPES.items():
            df[col] = df[col].astype(dtype)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    if len(df) > 0:
        n_ran = int(df["ran_in_cycle"].sum())
        n_matched_ran = int((df["has_fec_record"] & df["ran_in_cycle"]).sum())
    else:
        n_ran = n_matched_ran = 0

    logger.info(
        "Campaign stats written to %s: %d total rows, %d ran, %d with FEC record",
        output_path,
        len(df),
        n_ran,
        n_matched_ran,
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object, default: float | None = None) -> float | None:
    """Convert a value to float, returning default on failure.

    Handles None, NaN strings, empty strings, and real NaN values.
    """
    if value is None:
        return default
    try:
        import math

        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default
