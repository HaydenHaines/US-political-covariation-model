# DuckDB Domain Unification

**Date:** 2026-03-26
**Status:** Approved
**Scope:** Close the parquet bypass in `api/main.py`; establish typed domain contracts for all data entering DuckDB; make polling data queryable rather than file-parsed at request time.

---

## Problem

`api/main.py` reads six parquet files at startup outside DuckDB:

- `data/propagation/community_weights_state_hac.parquet`
- `data/propagation/community_weights_county_hac.parquet`
- `data/communities/type_assignments.parquet`
- `data/covariance/type_covariance.parquet`
- `data/communities/type_profiles.parquet`
- `data/models/ridge_model/ridge_county_priors.parquet`

The `/polls` endpoint parses a CSV at request time via `load_polls_with_notes()`.

The result: the API has two data sources (DuckDB + filesystem), no validation on what enters either, and no consistent pattern for adding new data domains (sabermetrics, candidate data).

---

## Design

### Domain Model

Four named data domains. Two active now; two reserved for future integration:

| Domain | Status | Persisted to DuckDB | Notes |
|---|---|---|---|
| **Model** | Active | Yes | Type scores, covariance, priors, ridge priors |
| **Polling** | Active | Yes | Poll rows, crosstabs, notes; CSV is source, DuckDB is queryable layer |
| **Candidate** | Reserved | — | Sabermetrics silo; declared but not implemented |
| **Runtime** | Structural | No | User what-ifs; always API request bodies, never persisted |

### DomainSpec

A pure data descriptor — no logic:

```python
@dataclass
class DomainSpec:
    name: str               # "model" | "polling" | "candidate" | "runtime"
    tables: list[str]       # DuckDB tables this domain owns
    description: str
    active: bool = True     # False = reserved, skip on build
    version_linked: bool = True  # whether rows carry version_id FK
```

### Directory Structure

```
src/db/
├── build_database.py          # orchestrator — calls each domain's ingest()
├── domains/
│   ├── __init__.py            # exports REGISTRY: list[DomainSpec]
│   ├── model.py               # DOMAIN_SPEC + ingest(db, version_id)
│   ├── polling.py             # DOMAIN_SPEC + ingest(db, cycle)
│   └── candidate.py           # DOMAIN_SPEC only (active=False)
```

---

## New DuckDB Tables

All version-linked tables carry `version_id VARCHAR` FK referencing `model_versions`.

### Model Domain

| Table | Key columns | Source |
|---|---|---|
| `type_scores` | county_fips, type_id, score FLOAT | `type_assignments.parquet` |
| `type_covariance` | type_i INT, type_j INT, value FLOAT | `type_covariance.parquet` (long-form J²) |
| `type_priors` | type_id INT, prior_value FLOAT | `type_profiles.parquet` |
| `ridge_county_priors` | county_fips, pred_dem_share FLOAT | `ridge_county_priors.parquet` |
| `hac_state_weights` | state_abbr, community_id INT, weight FLOAT | `community_weights_state_hac.parquet` |
| `hac_county_weights` | county_fips, community_id INT, weight FLOAT | `community_weights_county_hac.parquet` |

### Polling Domain

| Table | Key columns | Notes |
|---|---|---|
| `polls` | poll_id, race, geography, geo_level, dem_share, n_sample, date, pollster, cycle | Scalar poll rows |
| `poll_crosstabs` | poll_id FK, demographic_group, group_value, dem_share, n_sample | Per-poll demographic breakdowns |
| `poll_notes` | poll_id FK, note_type, value | Pollster quality flags, methodology notes |

`poll_crosstabs` is created on schema init; populated empty until crosstab data is available.

---

## Build Pipeline

`build_database.py --reset` runs stages in order:

```
1. build_core_tables()         # counties, model_versions (existing, unchanged)
2. ingest(model_domain)        # parquets → 6 type/covariance/weight tables
3. ingest(polling_domain)      # CSV → polls, poll_crosstabs, poll_notes
4. build_predictions()         # existing type-primary prediction pipeline
5. validate_contract()         # extended to cover new tables
```

Each `ingest()` validates source data against Pydantic schemas before writing. A validation failure aborts the build; no partial writes.

---

## Validation

### Build-time Pydantic schemas

**Model domain:**
- `TypeScoreRow`: `county_fips: str`, `type_id: int`, `score: float` (ge=0, le=1)
- `TypeCovarianceRow`: `type_i: int`, `type_j: int`, `value: float`; symmetric check on full matrix
- `TypePriorRow`: `type_id: int`, `prior_value: float` (ge=0, le=1)
- `RidgeCountyPriorRow`: `county_fips: str`, `pred_dem_share: float` (ge=0, le=1)

**Polling domain:**
- `PollIngestRow`: `race: str`, `geography: str`, `geo_level: Literal["state","county","district"]`, `dem_share: float` (ge=0, le=1), `n_sample: int` (gt=0), `date: str | None`, `cycle: str`

### Cross-compliance checks (post-ingest)

- `type_scores.county_fips ⊆ counties.county_fips`
- type_ids consistent across `type_scores`, `type_covariance`, `type_priors`
- `ridge_county_priors.county_fips ⊆ counties.county_fips`
- `polls.geography` for `geo_level="state"` ⊆ known state abbreviations

### Error handling

| Failure point | Behavior |
|---|---|
| Source parquet/CSV missing at build time | `DomainIngestionError(domain, path)` — build aborts |
| Row fails Pydantic validation | Build aborts; logs domain, source, field, offending value |
| Cross-compliance check fails | Build aborts with diff of mismatched FK values |
| DuckDB table missing at API startup | `RuntimeError` with domain and table name |

---

## API Changes

### `api/main.py` startup

The six `pd.read_parquet(...)` calls are replaced by SQL reads. Numpy arrays and dicts in `app.state` are reconstructed from DuckDB rows:

| `app.state` field | SQL source | Reconstruction |
|---|---|---|
| `type_scores` | `type_scores` | pivot(county_fips × type_id) → N×J array |
| `type_covariance` | `type_covariance` | pivot(type_i × type_j) → J×J array |
| `type_priors` | `type_priors` | sort by type_id → J-vector |
| `ridge_priors` | `ridge_county_priors` | dict[fips → float] |
| `state_weights` | `hac_state_weights` | pivot → DataFrame (existing shape) |
| `county_weights` | `hac_county_weights` | pivot → DataFrame (existing shape) |
| `type_county_fips` | `type_scores` | ordered list of county_fips |

The prediction pipeline (`predict_race`, Bayesian update) receives the same array shapes — the interface between pipeline and API does not change.

### `/polls` endpoint

Drops `load_polls_with_notes()`. Queries `polls` table with parameterized SQL. Same `PollRow` response shape.

---

## Testing

**`tests/test_db_builder.py` (extended):**
- Each domain's `ingest()` tested with minimal valid fixtures
- Cross-compliance violations (unknown county_fips, type_id gap) verified to abort build
- Symmetric matrix check tested with asymmetric covariance input

**`tests/test_api_contract.py` (extended):**
- `app.state` reconstruction: correct array shapes, no NaNs in priors, type count consistent across tables
- All six new tables present and non-empty in contract check

**`api/tests/` (updated):**
- `/polls` tests query in-memory `polls` table (CSV mock removed)
- `/forecast/poll` and `/forecast/polls` tests unchanged

**Out of scope:**
- Numpy reconstruction math (prediction pipeline's responsibility)
- Parquet parsing (assembly pipeline's responsibility)
- Pydantic field-level validation

---

## What Does Not Change

- Prediction pipeline interfaces (`predict_race`, `_forecast_poll_types`, `_forecast_poll_hac`)
- `api/models.py` response models (output contracts)
- `GET /forecast`, `POST /forecast/poll`, `POST /forecast/polls` response shapes
- `GET /polls` response shape
- HAC fallback logic (still loads from `app.state`; source changes, not shape)
