# Poll Data: Format and Usage

## File naming

Each election cycle has its own CSV:

```
data/polls/polls_2026.csv
data/polls/polls_2024.csv   # (if added for validation)
```

Load them with `src/assembly/ingest_polls.py`.

---

## Column definitions

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `race` | string | `2026 FL Senate` | Descriptive race name. Must match across rows for the same contest. |
| `geography` | string | `FL` | State abbreviation, or county FIPS code for sub-state polls (e.g. `12086`). |
| `geo_level` | string | `state` | One of: `state`, `county`, `district`. All current polls are `state`. |
| `dem_share` | float | `0.487` | Two-party Democratic fraction (range: 0–1). See calculation note below. |
| `n_sample` | integer | `800` | Effective sample size (not raw panel size if weighting applied). |
| `date` | string | `2026-02-15` | Poll field date in ISO format YYYY-MM-DD. Use midpoint of field window if multi-day. |
| `pollster` | string | `Quinnipiac University` | Polling firm name. |
| `notes` | string | `LV screen` | Optional context: screen type, methodology notes, caveats. |

---

## How to add a new poll

1. Open `data/polls/polls_2026.csv`.
2. Append a new row. All columns are required except `notes`.
3. Compute `dem_share` as the **two-party share** (see below).
4. Use the same `race` string used in existing rows for that contest (e.g. `2026 GA Senate`).

---

## dem_share: two-party share, not raw percentage

`dem_share` is always the Democratic fraction of the **two-party vote only** — third-party and undecided respondents are excluded.

**Formula:**

```
dem_share = D_pct / (D_pct + R_pct)
```

**Example 1:** Poll shows Harris 51%, Trump 45%, Other 4%

```
dem_share = 0.51 / (0.51 + 0.45) = 0.531
```

**Example 2:** Poll shows Dem 43%, Rep 49%, Undecided 8%

```
dem_share = 0.43 / (0.43 + 0.49) = 0.467
```

Do **not** enter the raw Democratic percentage (e.g. `0.51` when the two-party share is `0.531`). The model uses `dem_share` directly as the poll observation `y` — entering the raw share would bias all downstream estimates.

---

## Sigma (sampling noise)

The loader does not require a `sigma` column. Sampling noise is computed automatically from `dem_share` and `n_sample` by the `PollObservation.sigma` property:

```
sigma = sqrt(p * (1 - p) / n_sample)
```

For a poll with `dem_share=0.50` and `n_sample=800`, `sigma ≈ 0.0177` (±1.8 percentage points at 1 SD).

---

## Geography codes

- State abbreviations: `FL`, `GA`, `AL`
- County FIPS codes (5-digit): e.g. `12086` for Miami-Dade, `13121` for Fulton County GA
- Congressional district: e.g. `FL-10`, `GA-06` (not yet used)

Set `geo_level` to match: `state`, `county`, or `district`.

---

## Loading polls in Python

```python
from src.assembly.ingest_polls import load_polls, load_polls_with_crosstabs, list_races, polls_summary

# All 2026 polls
polls = load_polls("2026")

# Filter by race (substring match, case-insensitive)
fl_senate = load_polls("2026", race="FL Senate")

# Filter by state and recency
recent_ga = load_polls("2026", geography="GA", after="2026-02-01")

# Print summary table
polls_summary("2026")

# List all race names in the CSV
print(list_races("2026"))

# Load polls with crosstab composition data (for crosstab-adjusted W vectors)
polls, crosstabs = load_polls_with_crosstabs("2026")
# crosstabs is a dict keyed by "{race}|{geography}|{date}|{pollster}"
# Each value is a list of {"demographic_group", "group_value", "pct_of_sample"} dicts
# Pass directly to src.propagation.crosstab_w_builder.construct_w_row()
```

---

## Crosstab columns (`xt_*` convention)

Polls with demographic crosstab data carry additional columns prefixed with `xt_`.
These columns encode the **fraction of the poll sample** in each demographic group,
allowing the propagation model to construct poll-specific W vectors that weight
types with matching demographic profiles more heavily.

### Column naming: `xt_<group>_<value>`

| Column | Meaning | Example value |
|--------|---------|---------------|
| `xt_education_college` | Share of respondents with a college degree | `0.55` |
| `xt_education_noncollege` | Share without a college degree | `0.45` |
| `xt_race_white` | Share who identify as non-Hispanic white | `0.62` |
| `xt_race_black` | Share who identify as Black | `0.13` |
| `xt_race_hispanic` | Share who identify as Hispanic | `0.15` |
| `xt_race_asian` | Share who identify as Asian | `0.05` |
| `xt_urbanicity_urban` | Share in urban areas | `0.48` |
| `xt_urbanicity_rural` | Share in rural areas | `0.22` |
| `xt_age_senior` | Share aged 65+ | `0.28` |
| `xt_religion_evangelical` | Share identifying as evangelical Christian | `0.30` |

Values must be in `[0, 1]`.  Missing or empty cells are silently skipped.
Not all polls have crosstab data — the `xt_*` columns are entirely optional.
A CSV without any `xt_*` columns is fully backward-compatible.

### Worked example

A Quinnipiac poll of Georgia Senate 2026 that oversampled college-educated
voters might have this row:

```
race,geography,geo_level,dem_share,n_sample,date,pollster,notes,xt_education_college,xt_education_noncollege,xt_race_white
2026 GA Senate,GA,state,0.52,800,2026-04-01,Quinnipiac,grade=A,0.55,0.45,0.62
```

The `xt_education_college=0.55` signals that 55% of respondents are college-educated,
compared to ~35% in the Georgia population.  The propagation model detects this
delta (+0.20) and shifts the W vector toward types with high college-educated membership,
producing a more accurate inference about which types the poll actually sampled.

### Implementation notes

- `src/assembly/ingest_polls.py` ignores `xt_*` columns in `load_polls()` (backward compat).
  Use `load_polls_with_crosstabs()` to get both polls and parsed crosstab dicts.
- `src/db/domains/polling.py` ingests `xt_*` columns into the `poll_crosstabs` DuckDB table.
- `src/propagation/crosstab_w_builder.py` consumes crosstab data to adjust W vectors.
- The recognized dimensions and their mapping to type features are defined in
  `CROSSTAB_DIMENSION_MAP` in `crosstab_w_builder.py`.
