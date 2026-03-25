# National Expansion Research — S186

**Date:** 2026-03-24
**Status:** Research complete, ready for implementation

## Key Finding

The county-level production model expansion to 50 states is **significantly simpler than expected**. Most raw data files are already national — the fetchers just filter to FL/GA/AL. The main work is removing state filters, downloading ~350MB of per-state MEDSL files, and re-running the pipeline.

## Current Data Status

| Dataset | Scope | National? |
|---------|-------|-----------|
| MEDSL presidential county (.tab) | 2000-2024 | **Already national** — fetcher filters to 3 states |
| MEDSL Senate county (.tab) | 1976+ | **Already national** |
| Algara/Amlani governor (.Rdata) | 1865-2020 | **Already national** |
| FIPS county crosswalk | 3,235 counties | **Already national** |
| MEDSL 2022 governor (per-state ZIPs) | FL/GA/AL only | Need ~36 more states |
| MEDSL 2024 president (per-state ZIPs) | FL/GA/AL only | Need ~47 more states |
| Census/ACS/RCMS/IRS | FL/GA/AL only | Config change (loop over STATES dict) |
| GeoJSON | FL/GA/AL only | Remove state filter in build script |
| VEST precinct shapefiles | FL/GA/AL only | Tract model only — NOT needed for county |

## Implementation Phases

### Phase A: Data Layer (no model changes)
1. Remove state filter from `fetch_medsl_county_presidential.py`
2. Remove state filter from `fetch_medsl_county_senate.py` + Algara
3. Expand `fetch_2024_president.py` STATES to all 50 states
4. Expand `fetch_2022_governor.py` to all ~39 governor states
5. Re-run all demographic fetchers (ACS, census, RCMS, IRS) with expanded STATES

### Phase B: Shift Vectors
6. Expand `config/model.yaml` geography to all 50 states
7. Re-run `build_county_shifts_multiyear.py` — zero-fill for missing governor years already implemented
8. Validate: expect ~3,100 counties x 57 dims

### Phase C: Geospatial
9. Remove state filter in `build_county_geojson.py` → `counties-us.geojson`
10. Update `MapShell.tsx` to load new GeoJSON
11. Consider simplification tolerance increase (0.001 → 0.01) for file size

### Phase D: Retrain
12. Run `select_j` sweep on 3,100 counties — expect J=60-100
13. Run `run_type_discovery` with optimal J
14. Re-run describe_types, covariance, DB build
15. Validate holdout r > 0.80

### Phase E: Cleanup
16. Replace `_STATE_FIPS_TO_ABBR` hardcode in `build_database.py` with crosswalk lookup
17. Update model version metadata

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Governor cycle heterogeneity (odd-year states) | Medium | Zero-fill already implemented; ~13 states lose governor dims |
| GeoJSON file size (~16-20MB for all US) | Medium | Higher simplification tolerance or vector tiles |
| Alaska districts vs counties | Low | MEDSL data artifact; zero-fill handles it |
| CT planning regions (2022 reorganization) | Low | Known FIPS mismatch; affects ~8 counties |

## Scope Estimate

- **Files to change:** 15-20 Python, 1 TypeScript, 1 YAML
- **New data to download:** ~350MB (MEDSL per-state files)
- **Model retrain time:** ~45 minutes (J sweep + discovery + DB build)
- **Total engineering effort:** 1-2 sessions

## Key Insight

VEST data, TIGER tract shapefiles, and governor cycle complexity only affect the **tract-level model**. The county-level production model needs only config changes, filter removals, and MEDSL downloads. This is much simpler than initially estimated.
