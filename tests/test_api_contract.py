"""Integration tests: validate DuckDB->API->frontend contract.

These tests use the REAL wethervane.duckdb to catch pipeline/API mismatches.
Skip gracefully if the DB file doesn't exist (CI without data).
"""
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

DB_PATH = Path("data/wethervane.duckdb")

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(),
    reason="data/wethervane.duckdb not found — skip contract integration tests",
)


@pytest.fixture(scope="module")
def real_client():
    """TestClient backed by the real wethervane.duckdb."""
    from api.main import create_app

    app = create_app()  # uses real lifespan
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_db():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    yield con
    con.close()


# ── Schema tests ──────────────────────────────────────────────────────────

def test_duckdb_contract(real_db):
    """Required tables and columns exist in wethervane.duckdb."""
    from src.db.build_database import validate_contract

    errors = validate_contract(real_db)
    assert errors == [], f"Contract violations: {errors}"


# ── API response shape tests ──────────────────────────────────────────────

def test_super_types_response(real_client):
    resp = real_client.get("/api/v1/super-types")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0, "super-types must not be empty"
    for st in data:
        assert "super_type_id" in st
        assert "display_name" in st
        assert isinstance(st["display_name"], str)
        assert len(st["display_name"]) > 0


def test_counties_reference_valid_super_types(real_client):
    super_types = {st["super_type_id"] for st in real_client.get("/api/v1/super-types").json()}
    counties = real_client.get("/api/v1/counties").json()
    for c in counties:
        if c["super_type"] is not None:
            assert c["super_type"] in super_types, \
                f"County {c['county_fips']} has super_type={c['super_type']} not in super-types"


def test_forecast_has_state_abbr(real_client):
    rows = real_client.get("/api/v1/forecast").json()
    if not rows:
        pytest.skip("No forecast data")
    for r in rows:
        assert "state_abbr" in r
        assert isinstance(r["state_abbr"], str)
        assert len(r["state_abbr"]) == 2


def test_type_detail_has_dynamic_dicts(real_client):
    types = real_client.get("/api/v1/types").json()
    if not types:
        pytest.skip("No types in database")
    detail = real_client.get(f"/api/v1/types/{types[0]['type_id']}").json()
    assert isinstance(detail["demographics"], dict)
    if detail["shift_profile"] is not None:
        assert isinstance(detail["shift_profile"], dict)
        # Verify no non-numeric metadata leaked into shift_profile
        for key, val in detail["shift_profile"].items():
            assert isinstance(val, (int, float)), \
                f"shift_profile[{key}] is {type(val).__name__}, expected number"


# ── Cross-layer consistency ───────────────────────────────────────────────

def test_super_type_coverage(real_client):
    super_type_ids = {st["super_type_id"] for st in real_client.get("/api/v1/super-types").json()}
    county_super_types = {c["super_type"] for c in real_client.get("/api/v1/counties").json() if c["super_type"] is not None}
    assert county_super_types <= super_type_ids, \
        f"Counties reference super-types not in API: {county_super_types - super_type_ids}"
    unused = super_type_ids - county_super_types
    if unused:
        import warnings
        warnings.warn(f"Super-types with no counties: {unused}")


def test_health_reports_contract_status(real_client):
    resp = real_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "contract" in data
    assert data["contract"] in ("ok", "degraded")


def test_poll_crosstabs_in_contract(real_db):
    """poll_crosstabs table exists with required columns (may be empty)."""
    n = real_db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'poll_crosstabs'"
    ).fetchone()[0]
    assert n == 1, "poll_crosstabs table not found in wethervane.duckdb"

    cols = set(real_db.execute("SELECT * FROM poll_crosstabs LIMIT 0").fetchdf().columns)
    for required_col in ("poll_id", "demographic_group", "group_value", "pct_of_sample"):
        assert required_col in cols, f"poll_crosstabs missing column '{required_col}'"


def test_crosstab_w_row_differs_from_baseline(tmp_path):
    """A poll with xt_education_college=0.55 should produce a different W row
    than one without crosstabs, when meaningful type-demographic affinity exists.

    This test exercises the full path from construct_w_row() through the affinity
    index to confirm the crosstab signal propagates into the W vector.
    """
    import numpy as np
    from src.propagation.crosstab_w_builder import (
        build_affinity_index,
        compute_state_baseline_w,
        construct_w_row,
    )

    rng = np.random.default_rng(42)
    J = 10
    N = 20

    # Build synthetic type profiles with enough spread that affinity is non-trivial.
    type_profiles = _make_synthetic_type_profiles(J=J, rng=rng)
    county_demographics = _make_synthetic_county_demographics(N=N, rng=rng)

    affinity_index = build_affinity_index(type_profiles, county_demographics)

    # State baseline W: uniform population, all counties in one state
    type_scores = rng.uniform(0.0, 1.0, size=(N, J))
    county_populations = rng.integers(10_000, 500_000, size=N).astype(float)
    state_mask = np.ones(N, dtype=bool)
    w_baseline = compute_state_baseline_w(type_scores, county_populations, state_mask)

    # State demographic means: 35% college (realistic US state average)
    state_means = {k: float(np.average(county_demographics[col].to_numpy(), weights=county_demographics["pop_total"].to_numpy()))
                   for k, col in [("education_college", "pct_bachelors_plus")]
                   if col in county_demographics.columns}
    # Fill in remaining dimensions at their national means
    pop_weights = county_demographics["pop_total"].to_numpy(dtype=float)
    from src.propagation.crosstab_w_builder import CROSSTAB_DIMENSION_MAP
    for dim_key, feature in CROSSTAB_DIMENSION_MAP.items():
        if feature is not None and feature in county_demographics.columns and dim_key not in state_means:
            state_means[dim_key] = float(np.average(county_demographics[feature].to_numpy(), weights=pop_weights))

    # Poll with college oversample (55% vs ~35% state mean) — strong signal
    crosstabs_with_xt = [
        {"demographic_group": "education", "group_value": "college", "pct_of_sample": 0.55},
    ]
    w_with_xt = construct_w_row(crosstabs_with_xt, w_baseline, affinity_index, state_means)

    # Poll without crosstabs must return baseline unchanged
    w_without_xt = construct_w_row([], w_baseline, affinity_index, state_means)

    # The crosstab-adjusted W must differ from the baseline
    assert not np.allclose(w_with_xt, w_without_xt), (
        "W vector with xt_education_college=0.55 should differ from baseline W "
        "(poll college % deviates significantly from state mean)"
    )

    # Both must be valid probability vectors
    assert abs(w_with_xt.sum() - 1.0) < 1e-9
    assert abs(w_without_xt.sum() - 1.0) < 1e-9
    assert (w_with_xt >= 0.0).all()


def _make_synthetic_type_profiles(J: int, rng: "np.random.Generator") -> "pd.DataFrame":
    """Synthetic type_profiles for integration test helpers."""
    import numpy as np
    import pandas as pd

    return pd.DataFrame({
        "type_id":            np.arange(J),
        "pop_total":          rng.integers(50_000, 500_000, size=J).astype(float),
        "pct_bachelors_plus": np.linspace(0.15, 0.65, J),   # strong spread for signal
        "pct_white_nh":       rng.uniform(0.30, 0.90, size=J),
        "pct_black":          rng.uniform(0.02, 0.40, size=J),
        "pct_hispanic":       rng.uniform(0.02, 0.45, size=J),
        "pct_asian":          rng.uniform(0.01, 0.20, size=J),
        "log_pop_density":    rng.uniform(1.5, 4.5, size=J),
        "median_age":         rng.uniform(30.0, 55.0, size=J),
        "evangelical_share":  rng.uniform(0.05, 0.80, size=J),
    })


def _make_synthetic_county_demographics(N: int, rng: "np.random.Generator") -> "pd.DataFrame":
    """Synthetic county demographics for integration test helpers."""
    import numpy as np
    import pandas as pd

    return pd.DataFrame({
        "county_fips":        [f"{i:05d}" for i in range(N)],
        "pop_total":          rng.integers(5_000, 800_000, size=N).astype(float),
        "pct_bachelors_plus": rng.uniform(0.10, 0.60, size=N),
        "pct_white_nh":       rng.uniform(0.20, 0.95, size=N),
        "pct_black":          rng.uniform(0.01, 0.50, size=N),
        "pct_hispanic":       rng.uniform(0.01, 0.50, size=N),
        "pct_asian":          rng.uniform(0.01, 0.20, size=N),
        "log_pop_density":    rng.uniform(1.0, 5.5, size=N),
        "median_age":         rng.uniform(28.0, 60.0, size=N),
        "evangelical_share":  rng.uniform(0.03, 0.85, size=N),
    })


def test_app_state_reconstruction_shapes(tmp_path):
    """Verify app.state arrays have correct shapes after loading from DuckDB."""
    import duckdb
    import numpy as np
    from api.main import _load_type_data_from_db

    J = 4
    N = 3

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE type_scores (county_fips VARCHAR, type_id INTEGER, score FLOAT, version_id VARCHAR)
    """)
    for fips in ["12001", "12003", "13001"]:
        for j in range(J):
            con.execute("INSERT INTO type_scores VALUES (?,?,?,?)", [fips, j, 1.0/J, "v1"])

    con.execute("""
        CREATE TABLE type_covariance (type_i INTEGER, type_j INTEGER, value FLOAT, version_id VARCHAR)
    """)
    for i in range(J):
        for j in range(J):
            con.execute("INSERT INTO type_covariance VALUES (?,?,?,?)", [i, j, float(i==j)*0.01, "v1"])

    con.execute("""
        CREATE TABLE type_priors (type_id INTEGER, mean_dem_share FLOAT, version_id VARCHAR)
    """)
    for j in range(J):
        con.execute("INSERT INTO type_priors VALUES (?,?,?)", [j, 0.45, "v1"])

    scores, fips_list, covariance, priors = _load_type_data_from_db(con, "v1")
    assert scores.shape == (N, J)
    assert covariance.shape == (J, J)
    assert priors.shape == (J,)
    assert len(fips_list) == N
