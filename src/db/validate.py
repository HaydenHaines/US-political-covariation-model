"""Validation and reporting for wethervane.duckdb.

Contains the API-frontend contract validator, referential integrity
checks, and the row-count summary reporter.
"""
from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


def validate_contract(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Validate DuckDB matches the API-frontend contract.

    Returns a list of violation strings. Empty list = pass.
    See docs/superpowers/specs/2026-03-21-api-frontend-contract-design.md
    """
    errors: list[str] = []

    required = {
        "super_types": ["super_type_id", "display_name"],
        "types": ["type_id", "super_type_id", "display_name"],
        "county_type_assignments": ["county_fips", "dominant_type", "super_type"],
        "tract_type_assignments": ["tract_geoid", "dominant_type", "super_type"],
        "counties": ["county_fips", "state_abbr", "county_name", "total_votes_2024"],
        "type_scores": ["county_fips", "type_id", "score"],
        "type_priors": ["type_id", "mean_dem_share"],
        "polls": ["poll_id", "race", "geography", "dem_share"],
        # poll_crosstabs is required so the crosstab-adjusted W pipeline can
        # always query it — it will simply be empty when no crosstab data exists.
        "poll_crosstabs": ["poll_id", "demographic_group", "group_value", "pct_of_sample"],
    }

    optional = {
        "predictions": ["county_fips", "race", "pred_dem_share"],
        "tract_predictions": ["tract_geoid", "race", "forecast_mode", "pred_dem_share"],
    }

    def _check_table(table: str, columns: list[str], is_required: bool) -> None:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            if is_required:
                errors.append(f"MISSING TABLE: {table}")
            return
        actual_cols = set(con.execute(f'SELECT * FROM "{table}" LIMIT 0').fetchdf().columns)
        for col in columns:
            if col not in actual_cols:
                errors.append(f"MISSING COLUMN: {table}.{col}")

    for table, columns in required.items():
        _check_table(table, columns, is_required=True)
    for table, columns in optional.items():
        _check_table(table, columns, is_required=False)

    # Referential integrity (only if required tables exist)
    if not any("MISSING TABLE" in e for e in errors):
        orphans = con.execute("""
            SELECT DISTINCT cta.super_type
            FROM county_type_assignments cta
            LEFT JOIN super_types st ON cta.super_type = st.super_type_id
            WHERE st.super_type_id IS NULL AND cta.super_type IS NOT NULL
        """).fetchdf()
        if not orphans.empty:
            ids = orphans["super_type"].tolist()
            errors.append(f"ORPHAN super_type values in county_type_assignments: {ids}")

        orphan_types = con.execute("""
            SELECT DISTINCT cta.dominant_type
            FROM county_type_assignments cta
            LEFT JOIN types t ON cta.dominant_type = t.type_id
            WHERE t.type_id IS NULL AND cta.dominant_type IS NOT NULL
        """).fetchdf()
        if not orphan_types.empty:
            ids = orphan_types["dominant_type"].tolist()
            errors.append(f"ORPHAN dominant_type values in county_type_assignments: {ids}")

    return errors


def report_summary(con: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    """Print a row-count summary for every table and the model versions registry."""
    log.info("Database build complete: %s", db_path)
    print("\n=== wethervane.duckdb summary ===")
    for table in [
        "counties", "model_versions", "community_assignments", "type_assignments",
        "county_shifts", "predictions", "community_sigma", "community_profiles",
        "county_demographics", "types", "county_type_assignments", "tract_type_assignments",
        "tract_predictions",
        "super_types", "type_covariance", "demographics_interpolated",
        "type_scores", "type_priors", "ridge_county_priors", "polls", "poll_notes",
        "races",
    ]:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n:,} rows")
        except Exception as e:
            print(f"  {table}: ERROR -- {e}")

    print("\n=== Model Versions ===")
    rows = con.execute(
        "SELECT version_id, role, k, shift_type, vote_share_type, holdout_r FROM model_versions ORDER BY version_id"
    ).fetchall()
    print(f"  {'version_id':<45}  {'role':<20}  {'k':>4}  {'shift_type':<10}  {'holdout_r'}")
    for row in rows:
        vid, role, k, st, vst, hr = row
        print(f"  {str(vid):<45}  {str(role):<20}  {str(k):>4}  {str(st):<10}  {str(hr)}")


def validate_predictions(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Tier 1 model sanity checks — blocks the build if predictions are broken.

    These checks catch systematic model bugs like the type-compression issue
    (#139) where unpolled races collapsed to type means ~0.49. They run after
    predictions are ingested into DuckDB but before the database is finalized.

    Returns a list of error strings. Empty list = all checks passed.
    """
    errors: list[str] = []

    # Check if predictions table exists and has data
    try:
        n_preds = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE forecast_mode = 'local'"
        ).fetchone()[0]
    except Exception:
        # predictions table may not exist yet — skip validation
        log.warning("predictions table not found — skipping prediction validation")
        return errors

    if n_preds == 0:
        log.warning("No local-mode predictions found — skipping prediction validation")
        return errors

    # ── Compute vote-weighted state predictions for Senate races ──────────
    state_preds = con.execute("""
        SELECT
            p.race,
            c.state_abbr AS state,
            SUM(p.pred_dem_share * c.total_votes_2024) / SUM(c.total_votes_2024) AS pred_dem_share,
            COUNT(*) AS n_counties
        FROM predictions p
        JOIN counties c ON p.county_fips = c.county_fips
        WHERE p.forecast_mode = 'local'
          AND p.race LIKE '%Senate%'
          AND c.state_abbr = SPLIT_PART(p.race, ' ', 2)
        GROUP BY p.race, c.state_abbr
    """).fetchdf()

    if state_preds.empty:
        log.warning("No Senate predictions found — skipping prediction validation")
        return errors

    import numpy as np

    # ── Check 1: Prediction spread (detects type-compression) ─────────────
    pred_std = state_preds["pred_dem_share"].std()
    if pred_std < 0.05:
        errors.append(
            f"PREDICTION SPREAD: std={pred_std:.4f} < 0.05. "
            f"Range: {state_preds['pred_dem_share'].min():.3f} to "
            f"{state_preds['pred_dem_share'].max():.3f}. "
            f"Likely type-compression bug."
        )

    # ── Check 2: Safe D states above floor ────────────────────────────────
    safe_d = {"MA": 0.53, "IL": 0.53, "RI": 0.53, "DE": 0.53}
    for state, floor in safe_d.items():
        row = state_preds[state_preds["state"] == state]
        if not row.empty:
            pred = float(row.iloc[0]["pred_dem_share"])
            if pred < floor:
                errors.append(
                    f"SAFE D STATE: {state} predicted {pred:.3f} "
                    f"(D+{(pred-0.5)*200:.1f}pp), expected > {floor:.2f}"
                )

    # ── Check 3: Safe R states below ceiling ──────────────────────────────
    safe_r = {"WY": 0.35, "WV": 0.40, "OK": 0.40, "ID": 0.40}
    for state, ceiling in safe_r.items():
        row = state_preds[state_preds["state"] == state]
        if not row.empty:
            pred = float(row.iloc[0]["pred_dem_share"])
            if pred > ceiling:
                errors.append(
                    f"SAFE R STATE: {state} predicted {pred:.3f} "
                    f"(R+{(0.5-pred)*200:.1f}pp), expected < {ceiling:.2f}"
                )

    # ── Check 4: Both D and R predictions exist ──────────────────────────
    has_d = (state_preds["pred_dem_share"] > 0.55).any()
    has_r = (state_preds["pred_dem_share"] < 0.45).any()
    if not (has_d and has_r):
        errors.append(
            f"ONE-SIDED PREDICTIONS: D-leaning(>0.55)={has_d}, "
            f"R-leaning(<0.45)={has_r}. Model should predict both."
        )

    # ── Check 5: NJ canary ────────────────────────────────────────────────
    nj = state_preds[state_preds["state"] == "NJ"]
    if not nj.empty:
        nj_pred = float(nj.iloc[0]["pred_dem_share"])
        if nj_pred < 0.50:
            errors.append(
                f"NJ CANARY: NJ predicted {nj_pred:.3f} "
                f"(R+{(0.5-nj_pred)*200:.1f}pp). NJ is D+16 — structural bug."
            )

    # ── Check 6: Cross-state correlation with 2024 presidential ──────────
    # Approximate 2024 presidential Dem share by state
    pres_2024 = {
        "MA": 0.63, "RI": 0.60, "DE": 0.58, "IL": 0.57, "OR": 0.56,
        "NJ": 0.57, "CO": 0.56, "VA": 0.54, "NM": 0.54, "MN": 0.53,
        "NH": 0.52, "ME": 0.53, "MI": 0.49, "GA": 0.49, "NC": 0.48,
        "TX": 0.46, "IA": 0.44, "SC": 0.44, "AK": 0.43, "KS": 0.42,
        "MT": 0.40, "MS": 0.40, "LA": 0.40, "KY": 0.37, "AL": 0.37,
        "AR": 0.36, "TN": 0.36, "SD": 0.36, "NE": 0.40, "OK": 0.34,
        "ID": 0.33, "WV": 0.30, "WY": 0.27,
    }
    pres_vals, pred_vals = [], []
    for _, row in state_preds.iterrows():
        if row["state"] in pres_2024:
            pres_vals.append(pres_2024[row["state"]])
            pred_vals.append(row["pred_dem_share"])

    if len(pres_vals) >= 10:
        r = float(np.corrcoef(pres_vals, pred_vals)[0, 1])
        if r < 0.70:
            errors.append(
                f"CROSS-STATE CORRELATION: r={r:.3f} < 0.70 with 2024 presidential. "
                f"Model predictions don't track known partisan lean."
            )

    # ── Check 7: Minimum county count per race ───────────────────────────
    thin_races = state_preds[state_preds["n_counties"] < 3]
    if not thin_races.empty:
        races = thin_races["race"].tolist()
        errors.append(
            f"THIN RACES: {len(races)} race(s) with <3 counties: {races[:5]}. "
            f"Data pipeline may have failed."
        )

    # ── Check 8: Extreme margins (no prediction > 0.95 or < 0.05) ──────
    # A county-level prediction outside [0.05, 0.95] is almost certainly a
    # numerical artifact — even the most partisan counties rarely exceed
    # ~90% for one party in statewide races.
    extreme = state_preds[
        (state_preds["pred_dem_share"] > 0.95) | (state_preds["pred_dem_share"] < 0.05)
    ]
    if not extreme.empty:
        examples = extreme[["state", "pred_dem_share"]].head(5).to_dict("records")
        errors.append(
            f"EXTREME MARGINS: {len(extreme)} state(s) with predictions "
            f"outside [0.05, 0.95]: {examples}. "
            f"Likely numerical overflow or prior miscalibration."
        )

    # ── Tier 2: Soft checks (warnings only, do not block build) ─────────
    warnings: list[str] = []

    # ── Warning 1: Large partisan lean deviation without poll evidence ───
    # If a state's prediction deviates > 15pp from the 2024 presidential
    # result, and there's no poll data for that race, flag it.  Large shifts
    # from structural baseline without polling evidence suggest model drift.
    polled_states: set[str] = set()
    try:
        polled_rows = con.execute("""
            SELECT DISTINCT geography FROM polls
            WHERE geo_level = 'state' AND race LIKE 'Senate%'
        """).fetchdf()
        polled_states = set(polled_rows["geography"].tolist())
    except Exception:
        pass  # polls table may not exist — skip this check

    for _, row in state_preds.iterrows():
        st = row["state"]
        if st in pres_2024 and st not in polled_states:
            deviation = abs(row["pred_dem_share"] - pres_2024[st])
            if deviation > 0.15:
                warnings.append(
                    f"LEAN DEVIATION: {st} prediction {row['pred_dem_share']:.3f} "
                    f"deviates {deviation*100:.1f}pp from 2024 pres ({pres_2024[st]:.2f}) "
                    f"with no poll evidence."
                )

    # ── Warning 2: Unpolled races clustering ─────────────────────────────
    # If 10+ unpolled states have predictions within a 2pp band, it suggests
    # the model is defaulting to type means rather than capturing state-level
    # variation.  This is the "uniform collapse" signature from #139.
    unpolled = state_preds[~state_preds["state"].isin(polled_states)]
    if len(unpolled) >= 10:
        pred_range = unpolled["pred_dem_share"].max() - unpolled["pred_dem_share"].min()
        if pred_range < 0.02:
            warnings.append(
                f"UNPOLLED CLUSTERING: {len(unpolled)} unpolled states cluster "
                f"within {pred_range*100:.1f}pp range "
                f"[{unpolled['pred_dem_share'].min():.3f}, "
                f"{unpolled['pred_dem_share'].max():.3f}]. "
                f"Model may be collapsing to type means."
            )

    for w in warnings:
        log.warning("Prediction check WARNING: %s", w)

    if errors:
        log.error("Prediction validation FAILED (%d issues):", len(errors))
        for e in errors:
            log.error("  %s", e)
    else:
        log.info(
            "Prediction validation PASSED — %d states, std=%.4f, "
            "range=[%.3f, %.3f]",
            len(state_preds),
            pred_std,
            state_preds["pred_dem_share"].min(),
            state_preds["pred_dem_share"].max(),
        )
    if warnings:
        log.info("Prediction validation produced %d warning(s)", len(warnings))

    return errors


def validate_integrity(con: duckdb.DuckDBPyConnection) -> None:
    """Run contract + prediction validation and exit with status 1 on any violation."""
    errors = validate_contract(con)
    pred_errors = validate_predictions(con)
    errors.extend(pred_errors)
    if errors:
        for e in errors:
            log.error("VALIDATION FAILURE: %s", e)
        del con
        gc.collect()
        sys.exit(1)
    log.info("All validation passed (contract + predictions)")
