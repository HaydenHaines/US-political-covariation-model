# Modeling Box Runbook: Real Polls + National DB Rebuild
**Date:** 2026-03-26
**Context:** feat/forecasting branch has all code changes. This runbook covers what to execute on the modeling box to: pull real polls, rebuild the national DuckDB, and verify the API serves the Forecast tab correctly.

---

## Overview

The modeling box has the national model artifacts (J=100, 3,154 counties) that are gitignored. This machine is the execution environment. The workflow is:

```
git pull → preflight → scrape polls → run predictions → rebuild DB → test API
```

The output is `data/wethervane.duckdb` — the single artifact the API needs.

---

## Step 0 — Pull latest code

```bash
git pull origin feat/forecasting
```

If feat/forecasting has been merged to main first:
```bash
git checkout main && git pull
```

---

## Step 1 — Pre-flight checks

Verify the gitignored model artifacts are present before running anything.

```bash
python - <<'EOF'
from pathlib import Path
root = Path(".")
checks = [
    "data/communities/type_assignments.parquet",
    "data/covariance/type_covariance.parquet",
    "data/communities/type_profiles.parquet",
    "data/communities/super_types.parquet",
    "data/communities/county_type_assignments_full.parquet",
    "data/shifts/county_shifts_multiyear.parquet",
    "data/models/ridge_model/ridge_county_priors.parquet",
    "data/raw/fips_county_crosswalk.csv",
]
missing = [p for p in checks if not (root / p).exists()]
if missing:
    print("MISSING:")
    for m in missing: print(f"  {m}")
else:
    print("All artifacts present.")
EOF
```

**If `county_type_assignments_full.parquet` is missing:** run the type assignment pipeline to generate it. This file contains `county_fips, dominant_type, super_type` and all `type_N_score` columns. Check `src/discovery/run_type_discovery.py` or equivalent for the output path.

**If `type_covariance.parquet` is missing:** run `python -m src.covariance.construct_type_covariance`.

**If `fips_county_crosswalk.csv` is missing:**
```bash
uv run python scripts/fetch_fips_crosswalk.py
```

---

## Step 2 — Scrape real 2026 polls

The scraper targets Wikipedia + 270toWin for FL, GA, AL governor and Senate races.

```bash
# Dry run first — prints parsed polls without writing
uv run python scripts/scrape_2026_polls.py --dry-run
```

Review the dry-run output:
- Check per-race counts. Expect 5–20 polls per race by mid-2026.
- If a race has 0 polls, the Wikipedia page structure may have changed — check `RACE_CONFIG` candidate name lists in `scripts/scrape_2026_polls.py` and update if needed.
- Verify dem_share values look plausible (0.35–0.65 range for competitive races).

```bash
# Write to data/polls/polls_2026.csv
uv run python scripts/scrape_2026_polls.py
```

Verify:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/polls/polls_2026.csv')
print(df.groupby('race')[['dem_share','n_sample']].describe().round(3))
print(f'Total: {len(df)} polls')
"
```

---

## Step 3 — Generate 2026 predictions

Runs the Bayesian update (poll propagation through type covariance) for each race in `polls_2026.csv`. Also generates a national `baseline` race (no polls, pure structural prior).

```bash
uv run python -m src.prediction.predict_2026_types
```

Verify:
```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/predictions/county_predictions_2026_types.parquet')
print(df.groupby('race')['county_fips'].count())
print(f'\nPrediction range: [{df.pred_dem_share.min():.3f}, {df.pred_dem_share.max():.3f}]')
"
```

Expected output: one entry per race per county (3,154 counties × n_races + baseline).

---

## Step 4 — Rebuild DuckDB

**Important:** the database file is `data/wethervane.duckdb` (not `data/bedrock.duckdb` — that's the old name from the pilot).

```bash
uv run python src/db/build_database.py --reset
```

This will print a table summary at the end. Expected healthy output:

```
=== wethervane.duckdb summary ===
  counties:                3,154 rows     ← national
  model_versions:          1+ rows
  type_assignments:        10+ rows       ← HAC community stubs (can be low; types table is primary)
  county_type_assignments: 3,154 rows     ← dominant_type + super_type per county
  types:                   100 rows       ← J=100 fine types
  super_types:             5–20 rows      ← Ward HAC super-types
  predictions:             3,154 × n_races rows
  county_shifts:           3,154 rows
  type_covariance:         10,000 rows    ← 100×100 = 10,000 cells in long format
```

If `county_type_assignments` shows 0 rows, `county_type_assignments_full.parquet` is missing — see Step 1.

If `type_covariance` shows 0 rows, `type_covariance_long.parquet` is missing. Check if `data/covariance/type_covariance_long.parquet` exists (build_database.py needs the long format). If only the wide format exists, convert:

```bash
python - <<'EOF'
import pandas as pd
cov = pd.read_parquet("data/covariance/type_covariance.parquet")
long = cov.stack().reset_index()
long.columns = ["type_id_row", "type_id_col", "covariance"]
long.to_parquet("data/covariance/type_covariance_long.parquet", index=False)
print(f"Wrote {len(long)} rows")
EOF
```

---

## Step 5 — Test the API locally

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Watch startup logs. Healthy startup looks like:
```
INFO: Using model version: <version>
INFO: Loaded sigma matrix (10×10)       ← HAC K=10 communities (OK to be small)
INFO: Loaded type_scores: 3154 counties x 100 types
INFO: Loaded type_covariance: 100 x 100
INFO: Loaded Ridge county priors: 3106 counties
INFO: Contract status: ok
```

If `Contract status: degraded`, one of `super_types`, `types`, or `county_type_assignments` is missing from the DB. Re-check Step 4.

Run a quick smoke test:
```bash
# List available races
curl -s "http://localhost:8000/api/v1/forecast" | python -m json.tool | python -c "
import json,sys
rows=json.load(sys.stdin)
races=sorted(set(r['race'] for r in rows))
print('Races:', races)
print('Counties:', len(rows))
"

# Test a specific race
curl -s "http://localhost:8000/api/v1/forecast?race=2026+FL+Senate" | \
    python -m json.tool | head -30

# Test the type system
curl -s "http://localhost:8000/api/v1/types" | python -c "
import json,sys; t=json.load(sys.stdin); print(f'{len(t)} types loaded')
"
```

---

## Step 6 — Verify the Forecast tab locally

With the API running, open the frontend:
```bash
cd web && npm run dev
# Navigate to http://localhost:3000/forecast
```

Check:
1. State dropdown populates (FL, GA, AL visible)
2. Selecting a state pans the map to that state
3. County outlines for the selected state turn white
4. Recalculate colors the map in blue/red partisan scale
5. County table shows predictions sorted by Dem share

---

## Step 7 — Deploy (if applicable)

If the API is deployed on this machine (Fly.io / Railway / uvicorn directly):

```bash
# Restart the API service to pick up the new wethervane.duckdb
# (exact command depends on your deployment method)

# For a direct uvicorn process:
pkill -f "uvicorn api.main:app" && nohup uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# For Fly.io:
fly deploy
```

---

## Known Gotchas

**`bedrock.duckdb` vs `wethervane.duckdb`:**
The repo was renamed from Bedrock to WetherVane. The old pilot database is `data/bedrock.duckdb` and is the old FL+GA+AL HAC model. The API looks for `data/wethervane.duckdb`. Don't confuse them. The `WETHERVANE_DB_PATH` env var can override the path if needed.

**race label format:**
Race labels use spaces: `"2026 FL Senate"`, `"2026 GA Governor"`. The scraper outputs this format; the prediction pipeline uses it; the API serves it. The frontend parses state by finding the first 2-char all-caps token in the race label.

**`type_covariance_long.parquet` vs `type_covariance.parquet`:**
The API loads `type_covariance.parquet` (wide J×J matrix) at startup. `build_database.py` needs `type_covariance_long.parquet` (long format) for the DB. Both should exist; if the long format is missing, see the conversion snippet in Step 4.

**Prediction coverage:**
`predict_2026_types.py` generates predictions for all counties for every race in `polls_2026.csv`, plus a `baseline` race (no polls). Each county gets a row per race. So if polls cover FL Senate + GA Senate + FL Governor, predictions cover all 3,154 counties × 4 races = 12,616 rows.

**Type priors:**
The API uses `type_profiles.parquet → mean_dem_share` as type priors. If this column is missing, it falls back to 0.45 for all types. Verify `type_profiles.parquet` has `mean_dem_share`.

---

## Reference: Key File Paths

| File | Purpose | Gitignored |
|------|---------|------------|
| `data/wethervane.duckdb` | Main DB served by API | Yes |
| `data/communities/type_assignments.parquet` | J=100 soft scores (N×J) | Yes |
| `data/communities/county_type_assignments_full.parquet` | dominant_type, super_type per county | Yes |
| `data/communities/type_profiles.parquet` | Type demographic profiles | Yes |
| `data/communities/super_types.parquet` | Super-type definitions | Yes |
| `data/covariance/type_covariance.parquet` | J×J covariance (wide) | Yes |
| `data/covariance/type_covariance_long.parquet` | J×J covariance (long) | Yes |
| `data/shifts/county_shifts_multiyear.parquet` | County shift vectors | Yes |
| `data/models/ridge_model/ridge_county_priors.parquet` | Ridge+HGB county priors | No (committed) |
| `data/polls/polls_2026.csv` | Scraped 2026 polls | No (committed) |
| `data/predictions/county_predictions_2026_types.parquet` | Type-primary predictions | Yes |
| `data/raw/fips_county_crosswalk.csv` | FIPS → county names | No (committed) |
