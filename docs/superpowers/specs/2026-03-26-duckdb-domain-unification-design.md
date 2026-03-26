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

The `/polls` and `/forecast/polls` endpoints parse a CSV at request time via `load_polls_with_notes()`.

The result: the API has two data sources (DuckDB + filesystem), no validation on what enters either, and no consistent pattern for adding new data domains (sabermetrics, candidate data).

---

## Design

### Domain Model

Four named data domains. Two active now; two reserved for future integration:

| Domain | Status | Persisted to DuckDB | Version key |
|---|---|---|---|
| **Model** | Active | Yes | `version_id` FK → `model_versions` |
| **Polling** | Active | Yes | `cycle` VARCHAR (e.g. "2026") |
| **Candidate** | Reserved | — | TBD |
| **Runtime** | Structural | No | N/A — always API request bodies |

### DomainSpec

A pure data descriptor — no logic:

```python
@dataclass
class DomainSpec:
    name: str               # "model" | "polling" | "candidate" | "runtime"
    tables: list[str]       # DuckDB tables this domain owns
    description: str
    active: bool = True     # False = reserved, skip on build
    version_key: str = "version_id"  # column used to scope rows ("version_id" or "cycle")
```

Note: `version_key` replaces the original `version_linked` flag. It is documentation only — `ingest()` implementations use it to name the discriminator column; no framework-level logic reads it at runtime.

### Directory Structure

```
src/db/
├── build_database.py          # orchestrator — calls each domain's ingest()
│                              # NOTE: existing builder logic is reorganized into
│                              # named stage functions as part of this work
├── domains/
│   ├── __init__.py            # exports REGISTRY: list[DomainSpec]
│   ├── model.py               # DOMAIN_SPEC + ingest(db, version_id)
│   ├── polling.py             # DOMAIN_SPEC + ingest(db, cycle)
│   └── candidate.py           # DOMAIN_SPEC only (active=False)
```

**`build_database.py` refactor scope:** The existing builder executes ~15 ad-hoc inline steps. As part of this work, those steps are wrapped into named stage functions (`build_core_tables`, `build_predictions`, `validate_contract`). This is a mechanical reorganization — no behavioral change to existing logic.

---

## New DuckDB Tables

### Model Domain

All rows carry `version_id VARCHAR` FK referencing `model_versions`.

**Source parquet formats (confirmed from codebase):**

- `type_assignments.parquet` — wide format: `county_fips` + `type_0_score … type_{J-1}_score`. Ingestion melts to long form.
- `type_covariance.parquet` — J×J wide matrix written by `construct_type_covariance.py` as `pd.DataFrame(covariance_matrix)` with integer column indices 0…J-1. Ingestion melts to long form.
- `type_profiles.parquet` — row per type, includes `mean_dem_share` column.
- `ridge_county_priors.parquet` — columns: `county_fips`, `ridge_pred_dem_share` (confirmed).

| Table | Columns | Source |
|---|---|---|
| `type_scores` | `county_fips VARCHAR`, `type_id INT`, `score FLOAT`, `version_id VARCHAR` | melt from `type_assignments.parquet` |
| `type_covariance` | `type_i INT`, `type_j INT`, `value FLOAT`, `version_id VARCHAR` | melt from `type_covariance.parquet` |
| `type_priors` | `type_id INT`, `mean_dem_share FLOAT`, `version_id VARCHAR` | from `type_profiles.parquet` |
| `ridge_county_priors` | `county_fips VARCHAR`, `ridge_pred_dem_share FLOAT`, `version_id VARCHAR` | from `ridge_county_priors.parquet` |
| `hac_state_weights` | `state_abbr VARCHAR`, `community_id INT`, `weight FLOAT`, `version_id VARCHAR` | from `community_weights_state_hac.parquet` |
| `hac_county_weights` | `county_fips VARCHAR`, `community_id INT`, `weight FLOAT`, `version_id VARCHAR` | from `community_weights_county_hac.parquet` |

**Note on `type_covariance`:** The existing `build_database.py` already has a `type_covariance` table (built from `type_covariance_long.parquet`). This work replaces that path with the same melt logic applied directly to `type_covariance.parquet` (the wide square matrix written by the covariance pipeline). Column names in DuckDB (`type_i`, `type_j`, `value`) are imposed by ingestion, not inherited from the parquet.

### Polling Domain

Rows carry `cycle VARCHAR` (e.g. "2026") rather than `version_id`.

| Table | Columns | Notes |
|---|---|---|
| `polls` | `poll_id VARCHAR PK`, `race VARCHAR`, `geography VARCHAR`, `geo_level VARCHAR`, `dem_share FLOAT`, `n_sample INT`, `date VARCHAR`, `pollster VARCHAR`, `cycle VARCHAR` | Scalar poll rows |
| `poll_crosstabs` | `poll_id VARCHAR FK`, `demographic_group VARCHAR`, `group_value VARCHAR`, `dem_share FLOAT`, `n_sample INT` | Per-poll demographic breakdowns; created empty until crosstab data is available |
| `poll_notes` | `poll_id VARCHAR FK`, `note_type VARCHAR`, `note_value VARCHAR` | Pollster quality flags and methodology notes. Example: `note_type="grade", note_value="A"`. Types drawn from `load_polls_with_notes()` note keys. |

`poll_id` is a stable hash of `(race, geography, date, pollster, cycle)`.

---

## Build Pipeline

`build_database.py --reset` runs stages in order:

```
1. build_core_tables()         # counties, model_versions (existing logic, now in named function)
2. ingest(model_domain)        # parquets → 6 type/covariance/weight tables
3. ingest(polling_domain)      # CSV → polls, poll_crosstabs, poll_notes
4. build_predictions()         # existing type-primary prediction pipeline (now in named function)
5. validate_contract()         # extended to cover new tables (now in named function)
```

Each `ingest()` validates source data against Pydantic schemas before writing. A validation failure aborts the build; no partial writes.

---

## Validation

### Build-time Pydantic schemas

**Model domain:**
- `TypeScoreRow`: `county_fips: str`, `type_id: int` (ge=0), `score: float` (ge=0, le=1)
- `TypeCovarianceRow`: `type_i: int` (ge=0), `type_j: int` (ge=0), `value: float`; symmetric check on full matrix
- `TypePriorRow`: `type_id: int` (ge=0), `mean_dem_share: float` (ge=0, le=1)
- `RidgeCountyPriorRow`: `county_fips: str`, `ridge_pred_dem_share: float` (ge=0, le=1)

**Polling domain:**
- `PollIngestRow`: `race: str`, `geography: str`, `geo_level: Literal["state","county","district"]`, `dem_share: float` (ge=0, le=1), `n_sample: int` (gt=0), `date: str | None`, `cycle: str`

### Cross-compliance checks (post-ingest)

- `type_scores.county_fips ⊆ counties.county_fips` — verifies no phantom FIPS in type data (model covers FL+GA+AL ⊂ full counties table; every model FIPS must resolve to a known county)
- `type_ids consistent` — MAX(type_id) in `type_scores` == MAX(type_i) in `type_covariance` == MAX(type_id) in `type_priors`
- `type_ids zero-indexed and contiguous` — sorted(unique(type_id)) == list(range(J)); enforced so pivot reconstruction produces a contiguous 0…J-1 column order matching the array shape expected by `predict_race`
- `ridge_county_priors.county_fips ⊆ counties.county_fips`
- `polls.geography` for `geo_level="state"` ⊆ known US state abbreviations

### Error handling

| Failure point | Behavior |
|---|---|
| Source parquet/CSV missing at build time | `DomainIngestionError(domain, path)` — build aborts |
| Row fails Pydantic validation | Build aborts; logs domain, source, field, offending value |
| Cross-compliance check fails | Build aborts with diff of mismatched values |
| DuckDB table missing at API startup | `RuntimeError` with domain and table name |

---

## API Changes

### `api/main.py` startup

The six `pd.read_parquet(...)` calls are replaced by SQL reads. Numpy arrays and dicts in `app.state` are reconstructed from DuckDB rows:

| `app.state` field | SQL source | Reconstruction |
|---|---|---|
| `type_scores` | `type_scores` | `pivot(county_fips × type_id).sort_index(axis=1).values` → N×J array; column order is 0…J-1 (enforced by cross-compliance) |
| `type_county_fips` | `type_scores` | ordered list matching row order of pivot |
| `type_covariance` | `type_covariance` | `pivot(type_i × type_j).sort_index(axis=0).sort_index(axis=1).values` → J×J array |
| `type_priors` | `type_priors` | `sort_values("type_id")["mean_dem_share"].values` → J-vector |
| `ridge_priors` | `ridge_county_priors` | `dict(zip(county_fips, ridge_pred_dem_share))` |
| `state_weights` | `hac_state_weights` | pivot → DataFrame matching existing shape |
| `county_weights` | `hac_county_weights` | pivot → DataFrame matching existing shape |

The prediction pipeline (`predict_race`, Bayesian update) receives the same array shapes — the interface between pipeline and API does not change.

### `/polls` endpoint

Drops `load_polls_with_notes()`. Queries `polls` table with parameterized SQL. Same `PollRow` response shape.

### `/forecast/polls` endpoint

Also calls `load_polls_with_notes()` in `api/routers/forecast.py`. This endpoint is updated in the same pass: poll loading switches from CSV parse to DuckDB query. The weighted Bayesian update logic (`apply_all_weights`, `predict_race`) is unchanged.

---

## Testing

**`tests/test_db_builder.py` (extended):**
- Each domain's `ingest()` tested with minimal valid fixtures
- Cross-compliance violations (unknown county_fips, non-contiguous type_ids, asymmetric covariance) verified to abort build with correct error

**`tests/test_api_contract.py` (extended):**
- `app.state` reconstruction: correct array shapes, no NaNs in priors, J consistent across `type_scores`/`type_covariance`/`type_priors`
- All six new model domain tables present and non-empty
- `polls` table present

**`api/tests/` (updated):**
- `/polls` and `/forecast/polls` tests query in-memory `polls` table (CSV mock removed)
- `/forecast/poll` tests unchanged

**Out of scope:**
- Numpy reconstruction math (prediction pipeline's responsibility)
- Parquet file writing (assembly pipeline's responsibility)
- Pydantic field-level validation

---

## What Does Not Change

- Prediction pipeline interfaces (`predict_race`, `_forecast_poll_types`, `_forecast_poll_hac`)
- `api/models.py` response models (output contracts)
- `GET /forecast`, `POST /forecast/poll`, `POST /forecast/polls` response shapes
- `GET /polls` response shape
- HAC fallback logic (still loads from `app.state`; source changes, not shape)
