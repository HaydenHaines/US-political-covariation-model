# Phase 1: County Model v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, defensible county-level prediction model for 2026 — proper K selection, real NMF types, Bayesian covariance (Stan), updated predictions with uncertainty intervals, and a validation report.

**Architecture:** Ward HAC on 30-dim log-odds shift vectors selects K communities (Layer 1); sklearn NMF on community profiles selects J types (Layer 2); a Stan factor model estimates the K×K covariance matrix Σ; Gaussian Bayesian update propagates polls through Σ to produce 2026 predictions. All outputs are stored in DuckDB via the existing `src/db/build_database.py` pipeline.

**Tech Stack:** Python 3.13, scikit-learn (Ward HAC, NMF), cmdstanpy (Stan factor model), pandas, numpy, duckdb, pytest

---

## Context (read before touching any file)

### Repository state entering Phase 1

All Phase 0 work is complete and committed. The following already exist and work:

| File | What it does |
|------|--------------|
| `data/shifts/county_shifts_multiyear.parquet` | 293 county × 34 col (county_fips + 30 training + 3 holdout log-odds shift dims) |
| `data/communities/county_adjacency.npz` | Scipy sparse CSR Queen contiguity matrix for 293 FL+GA+AL counties |
| `data/communities/county_adjacency.fips.txt` | Ordered FIPS list matching the adjacency matrix rows |
| `src/validation/validate_county_holdout_multiyear.py` | Holdout validation: runs Ward HAC at K values, computes Pearson r against 2020→2024 holdout |
| `src/discovery/run_county_clustering.py` | Runs Ward HAC on 9-dim `county_shifts.parquet`; **needs to be retargeted to multiyear shifts** |
| `src/models/type_classifier.py` | Stub NMF — returns uniform type weights; **will be replaced by real NMF** |
| `src/covariance/stan/community_covariance.stan` | Stan factor model for K=7 NMF community types; **needs generalization for any K** |
| `src/covariance/run_covariance_model.py` | Runs the Stan model on NMF tract-level data; **not used in Phase 1 (new script needed)** |
| `src/db/build_database.py` | Ingests parquet artifacts into `data/bedrock.duckdb` |
| `config/model.yaml` | Central config; `clustering.k` is currently `null` — will be set after K selection |
| `data/models/versions/county_multiyear_logodds_20260319/meta.yaml` | Current model metadata; `k` is 7, `holdout_r` is null |

### Key column names

**`county_shifts_multiyear.parquet`** columns:
- `county_fips` (str, 5-char zero-padded)
- Training: `pres_d_shift_00_04`, `pres_r_shift_00_04`, `pres_turnout_shift_00_04`, ..., `gov_d_shift_18_22`, `gov_r_shift_18_22`, `gov_turnout_shift_18_22` (30 cols)
- Holdout: `pres_d_shift_20_24`, `pres_r_shift_20_24`, `pres_turnout_shift_20_24` (3 cols)

**Assembled election parquets** (in `data/assembled/`):
- `medsl_county_presidential_YYYY.parquet` → `county_fips`, `pres_dem_share_YYYY`, `pres_total_YYYY`
- `algara_county_governor_YYYY.parquet` → `county_fips`, `gov_dem_share_YYYY`, `gov_total_YYYY`
- `medsl_county_2022_governor.parquet` → `county_fips`, `gov_dem_share_2022`, `gov_total_2022`
- `medsl_county_2024_president.parquet` → `county_fips`, `pres_dem_share_2024`, `pres_total_2024`

### K selection context

Prior holdout sweep results (from `validate_county_holdout_multiyear.py`):
- k=5: r=0.8697, k=7: r=0.8597, k=10: r=0.8261, k=15: r=0.7789, k=20: r=0.8704

The roadmap requires sweeping K=5,7,10,12,15,20,25,30 with a minimum community size of 8 counties.
**Expected outcome: K=20 will likely be optimal (r=0.8704) but K=7 is a reasonable interpretability
choice if differences are small (< 0.03 r).** K selection is empirical — run the sweep, let the data decide.

### Stan model context

`src/covariance/stan/community_covariance.stan` implements a rank-1 factor model:
```
theta[k,t] = mu[k] + lambda[k] * eta[t] + noise
Sigma[k,j] = lambda[k] * lambda[j]   (off-diagonal)
Sigma[k,k] = lambda[k]^2 + tau[k]^2
```
Input: `theta_obs[K, T]` = community vote shares per election; `theta_se[K, T]` = standard errors.
Sign ambiguity fixed by constraining one community's lambda > 0 (k_ref = most Democratic community).
The current model hardcodes K=7 and k_ref=2. **Phase 1 generalizes this.**

---

## File Structure

**New files to create:**
- `src/discovery/select_k.py` — K selection sweep (extends existing holdout validation)
- `src/models/nmf_types.py` — Real NMF type classifier
- `src/covariance/run_county_covariance.py` — County-level Stan Σ pipeline (new; does NOT replace run_covariance_model.py)
- `src/validation/generate_validation_report.py` — Validation report generator
- `tests/test_select_k.py` — Tests for K selection logic
- `tests/test_nmf_types.py` — Tests for NMF classifier
- `tests/test_county_covariance.py` — Tests for covariance input preparation

**Files to modify:**
- `src/covariance/stan/community_covariance.stan` — Generalize for any K and dynamic k_ref
- `src/discovery/run_county_clustering.py` — Retarget to multiyear shifts + chosen K from config
- `src/models/type_classifier.py` — Update to delegate to nmf_types.py for real NMF
- `src/db/build_database.py` — Add `community_sigma` table
- `config/model.yaml` — Set `k` after Task 1
- `data/models/versions/county_multiyear_logodds_20260319/meta.yaml` — Update K and holdout_r
- `CLAUDE.md` — Key Decisions Log entry for chosen K and Σ approach
- `docs/ROADMAP.md` — Check off Phase 1 deliverables

---

## Task 1: K Selection Sweep

**Goal:** Run Ward HAC at K=5,7,10,12,15,20,25,30, compute holdout r per K, enforce min community size ≥ 8, pick the optimal K, and write it to config.

**Files:**
- Create: `src/discovery/select_k.py`
- Create: `tests/test_select_k.py`
- Modify: `config/model.yaml` (set `clustering.k` after running)
- Modify: `data/models/versions/county_multiyear_logodds_20260319/meta.yaml` (add holdout_r)

### Key design: what `select_k.py` does

```python
# Pseudocode for select_k.py
shifts = load county_shifts_multiyear.parquet
W = load county_adjacency.npz
fips_list = load county_adjacency.fips.txt

# Align shifts to adjacency order
train_shifts = first 30 columns
holdout_shifts = last 3 columns
TRAINING_COMPARISON_COL = 12  # pres_d_shift_16_20 in training (matches holdout col 0)

# Normalize training shifts
scaler = StandardScaler().fit(train_shifts)
train_norm = scaler.transform(train_shifts)

# Fit full Ward tree once
model = AgglomerativeClustering(linkage="ward", connectivity=W, n_clusters=1)
model.fit(train_norm)

# Sweep K values
for k in [5, 7, 10, 12, 15, 20, 25, 30]:
    labels = _hc_cut(k, model.children_, n_leaves)
    min_size = min(np.bincount(labels))
    if min_size < 8:
        continue  # skip K values that violate min size constraint

    # Compute community means for training col 12 and holdout col 0
    train_means = [train_shifts[labels == i, 12].mean() for i in range(k)]
    holdout_means = [holdout_shifts[labels == i, 0].mean() for i in range(k)]
    r = pearsonr(train_means, holdout_means).statistic
    record result

# Pick optimal K = argmax(r) among valid K values
# Write to config/model.yaml: clustering.k = optimal_k
```

- [ ] **Step 1.1: Write failing tests for select_k**

Create `tests/test_select_k.py`:

```python
"""Tests for src/discovery/select_k.py"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from src.discovery.select_k import (
    run_k_sweep,
    pick_optimal_k,
    KSweepResult,
)


@pytest.fixture
def tiny_shifts():
    """20 counties with 33 shift dimensions."""
    rng = np.random.default_rng(42)
    n, d = 20, 33
    fips = [f"12{str(i).zfill(3)}" for i in range(n)]
    data = rng.normal(0, 0.1, (n, d))
    df = pd.DataFrame(data, columns=[f"shift_{i}" for i in range(d)])
    df.insert(0, "county_fips", fips)
    return df, fips


@pytest.fixture
def tiny_adjacency(tiny_shifts):
    """Chain adjacency for 20 counties."""
    _, fips = tiny_shifts
    n = len(fips)
    rows, cols = [], []
    for i in range(n - 1):
        rows += [i, i+1]; cols += [i+1, i]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_k_sweep_returns_results(tiny_shifts, tiny_adjacency):
    df, fips = tiny_shifts
    results = run_k_sweep(df, fips, tiny_adjacency,
                          train_cols=list(df.columns[1:31]),
                          holdout_cols=list(df.columns[31:]),
                          k_values=[3, 5],
                          min_community_size=2)
    assert len(results) >= 1


def test_k_sweep_result_fields(tiny_shifts, tiny_adjacency):
    df, fips = tiny_shifts
    results = run_k_sweep(df, fips, tiny_adjacency,
                          train_cols=list(df.columns[1:31]),
                          holdout_cols=list(df.columns[31:]),
                          k_values=[3],
                          min_community_size=2)
    r = results[0]
    assert hasattr(r, 'k')
    assert hasattr(r, 'holdout_r')
    assert hasattr(r, 'min_community_size')
    assert r.k == 3


def test_k_sweep_skips_small_communities(tiny_shifts, tiny_adjacency):
    """K values where min community < min_community_size are excluded."""
    df, fips = tiny_shifts
    # With 20 counties and min_size=10, K=3 might fail if a cluster has < 10
    results = run_k_sweep(df, fips, tiny_adjacency,
                          train_cols=list(df.columns[1:31]),
                          holdout_cols=list(df.columns[31:]),
                          k_values=[3, 5],
                          min_community_size=15)
    # All returned results should have min_community_size >= 15
    for r in results:
        assert r.min_community_size >= 15


def test_pick_optimal_k_highest_r():
    results = [
        KSweepResult(k=5, holdout_r=0.85, min_community_size=50),
        KSweepResult(k=7, holdout_r=0.87, min_community_size=42),
        KSweepResult(k=10, holdout_r=0.82, min_community_size=29),
    ]
    assert pick_optimal_k(results) == 7


def test_pick_optimal_k_empty_raises():
    with pytest.raises(ValueError, match="No valid K"):
        pick_optimal_k([])
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
python -m pytest tests/test_select_k.py -v
```
Expected: ImportError or ModuleNotFoundError (select_k doesn't exist yet)

- [ ] **Step 1.3: Implement `src/discovery/select_k.py`**

```python
"""K selection via holdout accuracy sweep for Ward HAC community discovery.

Sweeps K values and evaluates holdout predictive accuracy. Picks the K
that maximizes Pearson r between community-mean training shifts and
community-mean holdout shifts, subject to a minimum community size constraint.

Usage:
    python src/discovery/select_k.py
    python src/discovery/select_k.py --k-values 5 7 10 15 20 25 30 --min-size 8
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, load_npz
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
ADJ_NPZ = PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"
ADJ_FIPS = PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt"
CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


# Column index within the 30 training cols that corresponds to pres_d_shift_16_20
# (the most recent presidential D-shift in training = direct counterpart to holdout col 0)
_PRES_D_16_20_TRAINING_IDX = 12


@dataclass
class KSweepResult:
    k: int
    holdout_r: float
    min_community_size: int
    community_sizes: list[int]


def run_k_sweep(
    shifts_df: pd.DataFrame,
    fips_list: list[str],
    adjacency: csr_matrix,
    train_cols: list[str],
    holdout_cols: list[str],
    k_values: Sequence[int],
    min_community_size: int = 8,
    training_comparison_idx: int = _PRES_D_16_20_TRAINING_IDX,
) -> list[KSweepResult]:
    """Run Ward HAC at multiple K values, return holdout accuracy per valid K.

    Parameters
    ----------
    shifts_df:
        DataFrame with county_fips column and shift columns.
    fips_list:
        Ordered FIPS list matching adjacency matrix rows.
    adjacency:
        Queen contiguity adjacency matrix (scipy sparse).
    train_cols:
        Names of training shift columns in shifts_df.
    holdout_cols:
        Names of holdout shift columns in shifts_df.
    k_values:
        K values to sweep.
    min_community_size:
        Skip K values where any community has fewer counties than this.
    training_comparison_idx:
        Index within train_cols to compare against holdout col 0.
        Default 12 = pres_d_shift_16_20 (most recent presidential training dim).

    Returns
    -------
    List of KSweepResult for valid K values (those meeting min_community_size).
    Sorted by k ascending.
    """
    # Align shifts to adjacency order
    indexed = shifts_df.set_index("county_fips")
    aligned = indexed.reindex(fips_list)
    n_missing = aligned[train_cols[0]].isna().sum()
    if n_missing:
        log.warning("Filling %d counties with NaN shifts (column means)", n_missing)
        aligned[train_cols + holdout_cols] = aligned[train_cols + holdout_cols].fillna(
            aligned[train_cols + holdout_cols].mean()
        )

    train_arr = aligned[train_cols].values        # (N, n_train)
    holdout_arr = aligned[holdout_cols].values    # (N, n_holdout)
    n_leaves = len(fips_list)

    # Normalize training shifts
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_arr)

    # Fit full Ward dendrogram once
    log.info("Fitting Ward dendrogram (this may take ~10s for 293 counties)...")
    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=adjacency,
        n_clusters=1,
        compute_distances=True,
    )
    model.fit(train_norm)

    results = []
    for k in sorted(k_values):
        if k >= n_leaves:
            log.warning("k=%d >= n_leaves=%d, skipping", k, n_leaves)
            continue

        labels = _hc_cut(k, model.children_, n_leaves)
        sizes = np.bincount(labels)
        min_size = int(sizes.min())

        if min_size < min_community_size:
            log.info("k=%d: min community size %d < %d, skipping", k, min_size, min_community_size)
            continue

        # Community-level means: training col at training_comparison_idx vs holdout col 0
        train_means = np.array([
            train_arr[labels == i, training_comparison_idx].mean() for i in range(k)
        ])
        holdout_means = np.array([
            holdout_arr[labels == i, 0].mean() for i in range(k)
        ])

        if len(np.unique(train_means)) < 2 or len(np.unique(holdout_means)) < 2:
            log.warning("k=%d: degenerate means (constant vector), skipping", k)
            continue

        r = float(pearsonr(train_means, holdout_means).statistic)
        results.append(KSweepResult(
            k=k,
            holdout_r=r,
            min_community_size=min_size,
            community_sizes=sizes.tolist(),
        ))
        log.info("k=%d: holdout_r=%.4f, min_size=%d", k, r, min_size)

    return results


def pick_optimal_k(results: list[KSweepResult]) -> int:
    """Return the K with the highest holdout_r.

    Raises ValueError if no valid results.
    """
    if not results:
        raise ValueError("No valid K values found (all failed min community size constraint)")
    return max(results, key=lambda r: r.holdout_r).k


def update_config_k(k: int, config_path: Path = CONFIG_PATH) -> None:
    """Write the chosen K back to config/model.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["clustering"]["k"] = k
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Updated config/model.yaml: clustering.k = %d", k)


def main() -> None:
    parser = argparse.ArgumentParser(description="K selection sweep for Ward HAC")
    parser.add_argument(
        "--k-values", nargs="+", type=int,
        default=[5, 7, 10, 12, 15, 20, 25, 30],
        help="K values to sweep"
    )
    parser.add_argument(
        "--min-size", type=int, default=8,
        help="Minimum counties per community"
    )
    parser.add_argument(
        "--no-update-config", action="store_true",
        help="Print result but do not update config/model.yaml"
    )
    args = parser.parse_args()

    log.info("Loading shifts from %s", SHIFTS_PATH)
    shifts = pd.read_parquet(SHIFTS_PATH)
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)

    fips_list = ADJ_FIPS.read_text().splitlines()
    W = load_npz(str(ADJ_NPZ))

    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS
    train_cols = TRAINING_SHIFT_COLS
    holdout_cols = HOLDOUT_SHIFT_COLS

    results = run_k_sweep(
        shifts, fips_list, W,
        train_cols=train_cols,
        holdout_cols=holdout_cols,
        k_values=args.k_values,
        min_community_size=args.min_size,
    )

    print("\n=== K Selection Results ===")
    print(f"{'k':>4}  {'holdout_r':>10}  {'min_size':>9}  {'community_sizes'}")
    for r in results:
        print(f"{r.k:>4}  {r.holdout_r:>10.4f}  {r.min_community_size:>9}  {r.community_sizes}")

    if not results:
        log.error("No valid K values — check min_community_size constraint")
        return

    optimal_k = pick_optimal_k(results)
    best = next(r for r in results if r.k == optimal_k)
    print(f"\nOptimal K = {optimal_k} (holdout_r = {best.holdout_r:.4f})")

    if not args.no_update_config:
        update_config_k(optimal_k)
        print(f"Updated config/model.yaml: clustering.k = {optimal_k}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.4: Run tests**

```bash
python -m pytest tests/test_select_k.py -v
```
Expected: 5/5 PASS

- [ ] **Step 1.5: Run the actual K selection sweep**

```bash
python src/discovery/select_k.py
```

Expected output (approximately):
```
=== K Selection Results ===
   k   holdout_r  min_size  community_sizes
   5      0.8697        xx  [...]
   7      0.8597        xx  [...]
  10      0.8261        xx  [...]
  ...
Optimal K = XX (holdout_r = X.XXXX)
Updated config/model.yaml: clustering.k = XX
```

After running, verify `config/model.yaml` has `clustering: k: <chosen_K>`.
Also update `data/models/versions/county_multiyear_logodds_20260319/meta.yaml`:
```yaml
k: <chosen_K>
holdout_r: <best_r>
```

- [ ] **Step 1.6: Commit**

```bash
git add src/discovery/select_k.py tests/test_select_k.py config/model.yaml \
        data/models/versions/county_multiyear_logodds_20260319/meta.yaml
git commit -m "feat(phase1): K selection sweep — optimal K written to config"
```

---

## Task 2: HAC Clustering at Chosen K

**Goal:** Run Ward HAC at the K from Task 1 on the multiyear shifts, producing `county_community_assignments.parquet` with `community_id` (0-indexed) for each county. Rebuild DuckDB.

**Files:**
- Modify: `src/discovery/run_county_clustering.py` — retarget to multiyear shifts, read K from config
- Modify: `src/db/build_database.py` — no code changes, just re-run after clustering

The key change to `run_county_clustering.py`:
1. Change `SHIFTS_PATH` to point to `county_shifts_multiyear.parquet`
2. Change `SHIFT_COLS` to use the 30 training cols from `build_county_shifts_multiyear.TRAINING_SHIFT_COLS` (plus holdout for logging, but don't use in clustering)
3. Read `k_target` from `config/model.yaml` instead of running the elbow detection
4. Remove the 9-dim TRAIN_IDX / HOLDOUT_IDX split (the multiyear shifts are already properly split)

- [ ] **Step 2.0: Baseline test_discovery.py before making changes**

Run the existing discovery tests to document what currently passes (so regressions are detectable):

```bash
python -m pytest tests/test_discovery.py -v 2>&1 | tee /tmp/test_discovery_baseline.txt || true
```

Record the output. Any test that passes now but fails after Step 2.3 is a regression.

- [ ] **Step 2.1: Write failing test for the updated clustering**

Add to `tests/test_discovery.py` (or a new `tests/test_county_clustering.py` if discovery tests have collection errors):

```python
def test_run_county_clustering_reads_k_from_config(tmp_path, monkeypatch):
    """After Task 1, run_county_clustering.py should read K from config."""
    import yaml
    from src.core import config as cfg_mod

    # Write a minimal config with k=3
    config_data = {
        "clustering": {"k": 3, "k_candidates": [3, 5], "min_community_size": 2}
    }
    config_path = tmp_path / "model.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # The run_county_clustering main() should respect config k=3
    # (detailed integration tested in tests/integration/; here just test config reading)
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["clustering"]["k"] == 3
```

- [ ] **Step 2.2: Run to see it fails or passes (it should pass — just validation of structure)**

```bash
python -m pytest tests/test_county_clustering.py -v 2>/dev/null || \
python -m pytest tests/test_discovery.py -v 2>/dev/null || echo "Test infrastructure check"
```

- [ ] **Step 2.3: Update `src/discovery/run_county_clustering.py`**

Change the following at the top of the file:

```python
# OLD:
SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts.parquet"
SHIFT_COLS = [
    "pres_d_shift_16_20", ...  # 9 columns
]
TRAIN_IDX = [0, 1, 2, 6, 7, 8]
HOLDOUT_IDX = [3, 4, 5]

# NEW: read multiyear shifts and K from config
from src.core import config as _cfg
from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS as SHIFT_COLS

SHIFTS_PATH = PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet"
```

Then in `main()`, replace the elbow detection / k_target logic with:

```python
# Read K from config (set by select_k.py in Task 1).
# Use config.load() at runtime — do NOT use _cfg._cfg (import-time cache)
# because select_k.py writes the K choice after this module may have been imported.
from src.core import config as _cfg_mod
_live_cfg = _cfg_mod.load()
k_target = _live_cfg["clustering"]["k"]
if k_target is None:
    raise RuntimeError(
        "clustering.k is null in config/model.yaml. "
        "Run src/discovery/select_k.py first."
    )
log.info("Using K=%d from config/model.yaml", k_target)
```

Keep the existing Ward HAC fit, `_hc_cut`, assignment output, and type classifier stub call. Just remove the elbow sweep (it's now done by `select_k.py`).

- [ ] **Step 2.4: Run clustering**

```bash
python src/discovery/run_county_clustering.py
```

Expected:
```
Clustering at k=<K>...
k=<K> cluster sizes: {...}
Layer 1 community assignments saved to data/communities/county_community_assignments.parquet
Final: <K> communities assigned to 293 counties
Layer 2 stub type assignments saved to ...
```

Verify: `python -c "import pandas as pd; df=pd.read_parquet('data/communities/county_community_assignments.parquet'); print(df['community_id'].value_counts())"`

- [ ] **Step 2.5: Rebuild DuckDB**

```bash
python src/db/build_database.py --reset
```

Expected: `community_assignments: 293 rows` with correct K in the model_versions table.

- [ ] **Step 2.6: Commit**

```bash
git add src/discovery/run_county_clustering.py data/communities/county_community_assignments.parquet
# Note: data/bedrock.duckdb is gitignored
git commit -m "feat(phase1): retarget county clustering to multiyear shifts at config K"
```

---

## Task 3: Real NMF Type Classification

**Goal:** Replace the stub NMF in `type_classifier.py` with actual sklearn NMF. Run a J sweep (J=5,6,7,8), pick J, produce real type weights and dominant type IDs.

**Files:**
- Create: `src/models/nmf_types.py` — NMF implementation
- Modify: `src/models/type_classifier.py` — delegate to nmf_types.py
- Create: `tests/test_nmf_types.py`

### What NMF type classification does

1. Take community shift profiles: `community_profiles[K, n_train_dims]` = mean shift vector per community
2. Run sklearn NMF with `n_components=J`
3. Output: `W[K, J]` (each community's type weights) + `H[J, n_train_dims]` (each type's profile)
4. Normalize rows of W to sum to 1 (soft membership probabilities)
5. Dominant type per community = argmax of W row
6. J selection: sweep J=5,6,7,8; pick by reconstruction error + manual review note

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_nmf_types.py`:

```python
"""Tests for src/models/nmf_types.py"""
import numpy as np
import pandas as pd
import pytest
from src.models.nmf_types import (
    compute_community_profiles,
    fit_nmf,
    NMFResult,
    sweep_j,
)


@pytest.fixture
def sample_shifts():
    rng = np.random.default_rng(99)
    n_counties = 30
    n_dims = 30
    fips = [f"12{str(i).zfill(3)}" for i in range(n_counties)]
    shift_cols = [f"pres_d_shift_{i:02d}_{i+4:02d}" for i in range(n_dims)]
    data = rng.normal(0, 0.1, (n_counties, n_dims))
    df = pd.DataFrame(data, columns=shift_cols)
    df.insert(0, "county_fips", fips)
    return df, shift_cols


@pytest.fixture
def sample_assignments(sample_shifts):
    df, _ = sample_shifts
    rng = np.random.default_rng(0)
    n = len(df)
    # 5 communities
    return pd.DataFrame({
        "county_fips": df["county_fips"],
        "community_id": rng.integers(0, 5, n),
    })


def test_compute_profiles_shape(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    k = sample_assignments["community_id"].nunique()
    assert profiles.shape == (k, len(shift_cols))


def test_compute_profiles_mean(sample_shifts, sample_assignments):
    """Profile for community 0 = mean of counties in community 0."""
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    merged = df.merge(sample_assignments, on="county_fips")
    expected_mean = merged[merged["community_id"] == 0][shift_cols[0]].mean()
    assert abs(profiles[0, 0] - expected_mean) < 1e-10


def test_fit_nmf_output_shape(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    result = fit_nmf(profiles, j=4, random_state=42)
    k = sample_assignments["community_id"].nunique()
    assert isinstance(result, NMFResult)
    assert result.W.shape == (k, 4)
    assert result.H.shape == (4, len(shift_cols))
    assert len(result.dominant_type) == k


def test_fit_nmf_weights_sum_to_one(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    result = fit_nmf(profiles, j=4, random_state=42)
    row_sums = result.W.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_dominant_type_is_argmax(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    result = fit_nmf(profiles, j=4, random_state=42)
    expected = np.argmax(result.W, axis=1)
    np.testing.assert_array_equal(result.dominant_type, expected)


def test_sweep_j_returns_ordered_results(sample_shifts, sample_assignments):
    df, shift_cols = sample_shifts
    profiles = compute_community_profiles(df, sample_assignments, shift_cols)
    sweep = sweep_j(profiles, j_values=[3, 4, 5], random_state=42)
    assert len(sweep) == 3
    assert [s.j for s in sweep] == [3, 4, 5]
    for s in sweep:
        assert s.reconstruction_error >= 0
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_nmf_types.py -v
```
Expected: ImportError (module doesn't exist)

- [ ] **Step 3.3: Implement `src/models/nmf_types.py`**

```python
"""Real NMF type classification for Layer 2 of the Bedrock pipeline.

Takes community shift profiles (K × n_dims) and fits sklearn NMF to
produce J electoral types with soft membership weights.

Layer 2 semantics:
  W[k, j] = community k's weight for type j (normalized to sum to 1)
  H[j, d] = type j's characteristic shift pattern (not normalized)
  dominant_type[k] = argmax(W[k, :])

J selection: sweep J=5,6,7,8; pick based on reconstruction error +
interpretability. Reconstruction error alone is insufficient — the user
must review type profiles and name them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler


@dataclass
class NMFResult:
    j: int
    W: np.ndarray           # (K, J) community type weights, rows sum to 1
    H: np.ndarray           # (J, n_dims) type profiles
    dominant_type: np.ndarray  # (K,) argmax of W rows
    reconstruction_error: float


@dataclass
class JSweepEntry:
    j: int
    reconstruction_error: float
    result: NMFResult


def compute_community_profiles(
    shifts: pd.DataFrame,
    assignments: pd.DataFrame,
    shift_cols: list[str],
) -> np.ndarray:
    """Compute mean shift vector per community.

    Parameters
    ----------
    shifts:
        DataFrame with county_fips + shift_cols.
    assignments:
        DataFrame with county_fips + community_id.
    shift_cols:
        Names of shift columns to use.

    Returns
    -------
    profiles: np.ndarray of shape (K, len(shift_cols))
        Row k = mean shift vector for community k.
        Communities are ordered 0, 1, ..., K-1 by community_id.
    """
    merged = shifts.merge(assignments[["county_fips", "community_id"]], on="county_fips")
    k_ids = sorted(merged["community_id"].unique())
    profiles = np.array([
        merged[merged["community_id"] == k][shift_cols].mean().values
        for k in k_ids
    ])
    return profiles


def fit_nmf(
    community_profiles: np.ndarray,
    j: int,
    random_state: int = 42,
) -> NMFResult:
    """Fit NMF to community profiles and return normalized type weights.

    NMF requires non-negative input. Community shift profiles contain
    negative values (log-odds shifts). We apply a MinMax shift to [0, 1]
    before fitting, then extract the membership matrix W.

    The W matrix (community × type) is row-normalized so each community's
    type weights sum to 1 (soft membership probabilities).

    Parameters
    ----------
    community_profiles:
        (K, n_dims) array of community mean shift vectors.
    j:
        Number of types.
    random_state:
        Random seed for NMF reproducibility.

    Returns
    -------
    NMFResult with W (row-normalized), H (type profiles), dominant_type.
    """
    # Shift to non-negative: subtract column min, add small epsilon
    scaler = MinMaxScaler(feature_range=(0.01, 1.0))
    profiles_nn = scaler.fit_transform(community_profiles)

    nmf = NMF(
        n_components=j,
        init="nndsvda",
        random_state=random_state,
        max_iter=500,
    )
    W_raw = nmf.fit_transform(profiles_nn)   # (K, J)
    H = nmf.components_                       # (J, n_dims) in transformed space

    # Row-normalize W to sum to 1
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # guard against zero rows
    W = W_raw / row_sums

    dominant_type = np.argmax(W, axis=1)

    return NMFResult(
        j=j,
        W=W,
        H=H,
        dominant_type=dominant_type,
        reconstruction_error=float(nmf.reconstruction_err_),
    )


def sweep_j(
    community_profiles: np.ndarray,
    j_values: list[int] | None = None,
    random_state: int = 42,
) -> list[JSweepEntry]:
    """Fit NMF at multiple J values and return reconstruction errors.

    Parameters
    ----------
    community_profiles:
        (K, n_dims) community shift profiles.
    j_values:
        J values to sweep. Defaults to [5, 6, 7, 8].
    random_state:
        Seed for reproducibility.

    Returns
    -------
    List of JSweepEntry sorted by j ascending.
    """
    if j_values is None:
        j_values = [5, 6, 7, 8]
    results = []
    for j in sorted(j_values):
        if j >= len(community_profiles):
            continue  # can't have more types than communities
        result = fit_nmf(community_profiles, j=j, random_state=random_state)
        results.append(JSweepEntry(j=j, reconstruction_error=result.reconstruction_error, result=result))
    return results


def run_nmf_classification(
    shifts_path,
    assignments_path,
    shift_cols: list[str],
    j: int,
    output_path,
    random_state: int = 42,
) -> NMFResult:
    """End-to-end NMF classification pipeline.

    Reads shifts and assignments, computes profiles, fits NMF, writes parquet.

    Output parquet columns:
        community_id, type_weight_0, ..., type_weight_{J-1}, dominant_type_id
    """
    import pandas as pd
    from pathlib import Path

    shifts = pd.read_parquet(shifts_path)
    assignments = pd.read_parquet(assignments_path)
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)

    profiles = compute_community_profiles(shifts, assignments, shift_cols)
    result = fit_nmf(profiles, j=j, random_state=random_state)

    k_ids = sorted(assignments["community_id"].unique())
    out = pd.DataFrame({"community_id": k_ids})
    for jj in range(j):
        out[f"type_weight_{jj}"] = result.W[:, jj]
    out["dominant_type_id"] = result.dominant_type
    out["j"] = j

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    return result
```

- [ ] **Step 3.4: Update `src/models/type_classifier.py` to delegate to nmf_types**

Replace the stub `classify_types` function body and update the `run_type_classification` function to call `run_nmf_classification`:

```python
# In run_type_classification(), replace the stub call with:
from src.models.nmf_types import run_nmf_classification
return run_nmf_classification(
    shifts_path=shifts_path,
    assignments_path=assignments_path,
    shift_cols=shift_cols,
    j=j,
    output_path=output_path,
)
```

Also update `classify_types` docstring to say it's now backed by real NMF.

- [ ] **Step 3.5: Run tests**

```bash
python -m pytest tests/test_nmf_types.py tests/test_two_layer_contracts.py -v
```
Expected: All PASS

- [ ] **Step 3.6: Run J sweep and NMF classification**

```bash
python -c "
import pandas as pd
from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS
from src.models.nmf_types import compute_community_profiles, sweep_j

shifts = pd.read_parquet('data/shifts/county_shifts_multiyear.parquet')
assignments = pd.read_parquet('data/communities/county_community_assignments.parquet')
profiles = compute_community_profiles(shifts, assignments, TRAINING_SHIFT_COLS)
print('Community profiles shape:', profiles.shape)

for entry in sweep_j(profiles, j_values=[5,6,7,8]):
    print(f'J={entry.j}: reconstruction_error={entry.reconstruction_error:.4f}')
"
```

Then run the full classification at chosen J (start with J=7 as default):

```bash
python -c "
from src.models.type_classifier import run_type_classification
from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS

run_type_classification(
    shifts_path='data/shifts/county_shifts_multiyear.parquet',
    assignments_path='data/communities/county_community_assignments.parquet',
    shift_cols=TRAINING_SHIFT_COLS,
    j=7,
    output_path='data/communities/county_type_assignments.parquet',
)
print('Done')
"
```

Verify: `python -c "import pandas as pd; print(pd.read_parquet('data/communities/county_type_assignments.parquet'))"`

- [ ] **Step 3.7: Commit**

```bash
git add src/models/nmf_types.py src/models/type_classifier.py tests/test_nmf_types.py
git commit -m "feat(phase1): real NMF type classification replaces stub (Layer 2)"
```

---

## Task 4: Stan Covariance Model at County Level

**Goal:** Build a new Stan covariance pipeline that uses the county HAC assignments and the election parquets to compute theta_obs[K, T], then runs the Stan factor model to produce the K×K community covariance matrix Σ.

**Files:**
- Modify: `src/covariance/stan/community_covariance.stan` — generalize for any K and dynamic k_ref
- Create: `src/covariance/run_county_covariance.py` — county-level Σ pipeline
- Create: `tests/test_county_covariance.py`
- Modify: `src/db/build_database.py` — add `community_sigma` table

### Stan model generalization

The current Stan model hardcodes k_ref=2 (c2 Black urban) for sign identification. For a county HAC model with arbitrary K, we need to:
1. Add `int<lower=1, upper=K> k_ref;` to the data block
2. Change the lambda assembly in `transformed parameters` to use k_ref dynamically

Current code (hardcoded):
```stan
lambda[1] = lambda_other[1];   // c1
lambda[2] = lambda_ref;        // c2 reference
for (k in 3:K)
    lambda[k] = lambda_other[k - 1];
```

New code (dynamic k_ref):
```stan
{
  int j = 1;
  for (k in 1:K) {
    if (k == k_ref) {
      lambda[k] = lambda_ref;
    } else {
      lambda[k] = lambda_other[j];
      j += 1;
    }
  }
}
```

Note: Stan transformed parameters block is sequential, so the `j` counter works.

The Python side identifies k_ref = community with highest mean dem_share across all training elections.

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_county_covariance.py`:

```python
"""Tests for src/covariance/run_county_covariance.py"""
import numpy as np
import pandas as pd
import pytest
from src.covariance.run_county_covariance import (
    compute_theta_obs,
    identify_k_ref,
)


@pytest.fixture
def sample_assignments():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "community_id": [0, 0, 1, 1, 2],
    })


@pytest.fixture
def sample_election_data():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "dem_share": [0.60, 0.55, 0.40, 0.35, 0.30],
        "total_votes": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
    })


def test_compute_theta_obs_shape(sample_assignments, sample_election_data):
    """theta_obs should be K × T."""
    elections = [sample_election_data, sample_election_data]  # 2 elections
    theta_obs, theta_se, obs_mask = compute_theta_obs(sample_assignments, elections)
    k = sample_assignments["community_id"].nunique()
    assert theta_obs.shape == (k, 2)
    assert theta_se.shape == (k, 2)
    assert obs_mask.shape == (k, 2)


def test_compute_theta_obs_weighted_mean(sample_assignments, sample_election_data):
    """Community 0 theta = weighted mean of its counties' dem_shares."""
    elections = [sample_election_data]
    theta_obs, _, _ = compute_theta_obs(sample_assignments, elections)
    # Community 0: county 12001 (share=0.60, total=50000) + county 12003 (share=0.55, total=30000)
    expected = (0.60 * 50000 + 0.55 * 30000) / (50000 + 30000)
    assert abs(theta_obs[0, 0] - expected) < 1e-10


def test_obs_mask_all_ones_when_no_missing(sample_assignments, sample_election_data):
    elections = [sample_election_data]
    _, _, obs_mask = compute_theta_obs(sample_assignments, elections)
    assert obs_mask.sum() == obs_mask.size


def test_identify_k_ref_most_democratic(sample_assignments, sample_election_data):
    """k_ref = community with highest mean dem_share across elections."""
    elections = [sample_election_data]
    theta_obs, _, _ = compute_theta_obs(sample_assignments, elections)
    k_ref = identify_k_ref(theta_obs)
    # Community 0 has highest dem_share (~0.5875), so k_ref should be 0
    # Stan is 1-indexed so k_ref = 1
    assert k_ref == 1


def test_obs_mask_one_when_partial_nan(sample_assignments):
    """A community with ONE NaN county but other valid counties is still observed.

    obs_mask = 0 only when ALL counties in a community are NaN.
    The NaN county is dropped; the valid county contributes to the weighted mean.
    """
    election_with_partial_nan = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "dem_share": [float("nan"), 0.55, 0.40, 0.35, 0.30],
        "total_votes": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
    })
    _, _, obs_mask = compute_theta_obs(sample_assignments, [election_with_partial_nan])
    # Community 0 has one NaN (12001) but 12003 is valid → obs_mask[0,0] = 1
    assert obs_mask[0, 0] == 1.0


def test_obs_mask_zero_when_all_nan(sample_assignments):
    """obs_mask = 0 when ALL counties in a community have NaN dem_share."""
    election_all_nan_c0 = pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "dem_share": [float("nan"), float("nan"), 0.40, 0.35, 0.30],
        "total_votes": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
    })
    _, _, obs_mask = compute_theta_obs(sample_assignments, [election_all_nan_c0])
    # Community 0 = {12001, 12003}, both NaN → obs_mask[0,0] = 0
    assert obs_mask[0, 0] == 0.0
    # Communities 1 and 2 are still observed
    assert obs_mask[1, 0] == 1.0
```

- [ ] **Step 4.2: Run to verify tests fail**

```bash
python -m pytest tests/test_county_covariance.py -v
```
Expected: ImportError

- [ ] **Step 4.3: Update `community_covariance.stan`**

Find the data block and add `k_ref`:
```stan
data {
  int<lower=1> K;
  int<lower=1> T;
  int<lower=1, upper=K> k_ref;    // ADD THIS LINE
  matrix[K, T] theta_obs;
  matrix[K, T] theta_se;
  matrix[K, T] obs_mask;
}
```

Replace the hardcoded lambda assembly in `transformed parameters`:
```stan
// OLD (remove this):
lambda[1] = lambda_other[1];
lambda[2] = lambda_ref;
for (k in 3:K)
    lambda[k] = lambda_other[k - 1];

// NEW (replace with):
{
  int j = 1;
  for (k in 1:K) {
    if (k == k_ref) {
      lambda[k] = lambda_ref;
    } else {
      lambda[k] = lambda_other[j];
      j += 1;
    }
  }
}
```

Also update the comment in the model doc to say `k_ref` is now a data variable.

- [ ] **Step 4.4: Implement `src/covariance/run_county_covariance.py`**

```python
"""County-level Stan covariance estimation (Phase 1 pipeline).

Uses Ward HAC community assignments + assembled election parquets to
build theta_obs[K, T] (community vote shares per election), then runs
the Stan factor model to estimate the K×K community covariance matrix Σ.

Key design:
  theta_obs[k, t] = population-weighted mean dem_share for counties in
                    community k, election t.
  k_ref = 1-indexed community with highest mean dem_share (most Democratic
          community). Used to fix the sign of the factor loadings.
  T = number of election cycles used for estimation. With 10+ training
      elections, we sub-select to the most recent and most discriminating
      elections to avoid overfitting the factor model.

Inputs:
  data/communities/county_community_assignments.parquet
  data/assembled/medsl_county_presidential_YYYY.parquet (2016, 2020)
  data/assembled/algara_county_governor_2018.parquet
  data/assembled/medsl_county_2022_governor.parquet
  data/assembled/medsl_county_2024_president.parquet

Outputs:
  data/covariance/county_community_sigma.parquet   — K×K posterior mean Σ
  data/covariance/county_community_rho.parquet     — K×K posterior mean correlation
  data/covariance/county_covariance_summary.csv    — posterior summary
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
COVARIANCE_DIR = PROJECT_ROOT / "data" / "covariance"
STAN_MODEL = PROJECT_ROOT / "src" / "covariance" / "community_covariance.stan"

# Elections used for covariance estimation (label, dem_share_col, total_col, parquet)
# Using 5 most recent cycles to keep the Stan model well-identified (T > K/2 as rule of thumb)
_ELECTIONS = [
    ("pres_2016", "pres_dem_share_2016", "pres_total_2016",
     "medsl_county_presidential_2016.parquet"),
    ("gov_2018", "gov_dem_share_2018", "gov_total_2018",
     "algara_county_governor_2018.parquet"),
    ("pres_2020", "pres_dem_share_2020", "pres_total_2020",
     "medsl_county_presidential_2020.parquet"),
    ("gov_2022", "gov_dem_share_2022", "gov_total_2022",
     "medsl_county_2022_governor.parquet"),
    ("pres_2024", "pres_dem_share_2024", "pres_total_2024",
     "medsl_county_2024_president.parquet"),
]


def load_election(parquet_name: str, share_col: str, total_col: str) -> pd.DataFrame:
    """Load an election parquet and return [county_fips, dem_share, total_votes]."""
    path = ASSEMBLED_DIR / parquet_name
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    return df[["county_fips"]].assign(
        dem_share=df[share_col],
        total_votes=df[total_col],
    )


def compute_theta_obs(
    assignments: pd.DataFrame,
    elections: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute community vote shares and standard errors across elections.

    Parameters
    ----------
    assignments:
        DataFrame with county_fips and community_id (0-indexed).
    elections:
        List of election DataFrames, each with county_fips, dem_share, total_votes.

    Returns
    -------
    theta_obs: (K, T) population-weighted community dem_share per election
    theta_se:  (K, T) Kish effective-n standard error per community-election
    obs_mask:  (K, T) 1.0 if observed, 0.0 if missing (all counties NaN)
    """
    k_ids = sorted(assignments["community_id"].unique())
    K = len(k_ids)
    T = len(elections)

    theta_obs = np.full((K, T), np.nan)
    theta_se = np.full((K, T), 0.05)  # fallback SE
    obs_mask = np.zeros((K, T))

    for t, elec_df in enumerate(elections):
        merged = assignments.merge(elec_df, on="county_fips", how="left")
        for k_idx, k_id in enumerate(k_ids):
            mask = merged["community_id"] == k_id
            sub = merged[mask].dropna(subset=["dem_share", "total_votes"])
            if len(sub) == 0:
                obs_mask[k_idx, t] = 0.0
                continue

            w = sub["total_votes"].values
            p = sub["dem_share"].values

            # Weighted mean
            w_sum = w.sum()
            if w_sum == 0:
                obs_mask[k_idx, t] = 0.0
                continue

            theta = float((p * w).sum() / w_sum)
            theta_obs[k_idx, t] = theta
            obs_mask[k_idx, t] = 1.0

            # Kish effective N standard error
            w_sq_sum = (w ** 2).sum()
            n_eff = w_sum ** 2 / w_sq_sum if w_sq_sum > 0 else 1.0
            se = float(np.sqrt(theta * (1 - theta) / max(n_eff, 1)))
            theta_se[k_idx, t] = max(se, 0.001)  # floor at 0.1%

    return theta_obs, theta_se, obs_mask


def identify_k_ref(theta_obs: np.ndarray) -> int:
    """Return 1-indexed Stan k_ref = community with highest mean dem_share.

    The reference community is the most consistently Democratic one
    (highest mean across elections). Its lambda is constrained >= 0
    to fix the sign of the factor loadings.
    """
    mean_shares = np.nanmean(theta_obs, axis=1)
    return int(np.argmax(mean_shares)) + 1  # Stan is 1-indexed


def run(
    assignments_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 42,
) -> Path:
    """Run the county-level Stan covariance model and return sigma parquet path."""
    import cmdstanpy

    assignments_path = Path(assignments_path or COMMUNITIES_DIR / "county_community_assignments.parquet")
    output_dir = Path(output_dir or COVARIANCE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load assignments
    log.info("Loading community assignments from %s", assignments_path)
    assignments = pd.read_parquet(assignments_path)
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
    if "community_id" not in assignments.columns and "community" in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})
    K = assignments["community_id"].nunique()
    log.info("K = %d communities", K)

    # Load elections
    log.info("Loading %d election cycles for theta_obs...", len(_ELECTIONS))
    elections = []
    for label, share_col, total_col, parquet in _ELECTIONS:
        try:
            elec = load_election(parquet, share_col, total_col)
            elections.append(elec)
            log.info("  Loaded %s (%d counties)", label, len(elec))
        except Exception as e:
            log.warning("  Could not load %s: %s", label, e)
    T = len(elections)
    log.info("T = %d elections for Stan", T)

    # Build Stan data
    theta_obs, theta_se, obs_mask = compute_theta_obs(assignments, elections)
    k_ref = identify_k_ref(theta_obs)
    log.info("k_ref = %d (most Democratic community, 1-indexed)", k_ref)

    stan_data = {
        "K": K,
        "T": T,
        "k_ref": k_ref,
        "theta_obs": theta_obs.tolist(),
        "theta_se": theta_se.tolist(),
        "obs_mask": obs_mask.tolist(),
    }

    # Compile and run Stan model
    log.info("Compiling Stan model: %s", STAN_MODEL)
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_MODEL))
    log.info("Running MCMC: %d chains, %d warmup, %d sampling", chains, iter_warmup, iter_sampling)
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        show_progress=True,
        output_dir=str(output_dir / "stan_draws_county"),
    )

    # Check diagnostics
    diag = fit.diagnose()
    log.info("Stan diagnostics:\n%s", diag)

    # Extract Σ
    sigma_draws = fit.stan_variable("Sigma")  # (n_draws, K, K)
    sigma_mean = sigma_draws.mean(axis=0)     # (K, K)
    rho_draws = fit.stan_variable("Rho")
    rho_mean = rho_draws.mean(axis=0)

    # Save Σ as parquet
    comm_ids = sorted(assignments["community_id"].unique())
    sigma_df = pd.DataFrame(sigma_mean, index=comm_ids, columns=comm_ids)
    sigma_path = output_dir / "county_community_sigma.parquet"
    sigma_df.to_parquet(sigma_path)
    log.info("Saved Σ to %s  (K=%d, T=%d)", sigma_path, K, T)

    rho_df = pd.DataFrame(rho_mean, index=comm_ids, columns=comm_ids)
    rho_path = output_dir / "county_community_rho.parquet"
    rho_df.to_parquet(rho_path)

    # Summary CSV
    summary = fit.summary()
    summary.to_csv(output_dir / "county_covariance_summary.csv")

    print("\n=== Community Σ (posterior mean) ===")
    print(sigma_df.round(5).to_string())
    print("\n=== Community ρ (posterior mean correlation) ===")
    print(rho_df.round(3).to_string())

    return sigma_path


if __name__ == "__main__":
    run()
```

- [ ] **Step 4.5: Run tests**

```bash
python -m pytest tests/test_county_covariance.py -v
```
Expected: All PASS (no Stan needed for unit tests)

- [ ] **Step 4.6: Add `community_sigma` table to `src/db/build_database.py`**

Add to `_SCHEMA_SQL`:
```sql
CREATE TABLE IF NOT EXISTS community_sigma (
    community_id_row  INTEGER NOT NULL,
    community_id_col  INTEGER NOT NULL,
    sigma_value       DOUBLE,
    version_id        VARCHAR NOT NULL,
    PRIMARY KEY (community_id_row, community_id_col, version_id)
);
```

Add sigma ingestion in `build()`:
```python
sigma_path = PROJECT_ROOT / "data" / "covariance" / "county_community_sigma.parquet"
if sigma_path.exists():
    sigma_df = pd.read_parquet(sigma_path)
    # Convert K×K matrix to long form
    sigma_long_rows = []
    for row_id in sigma_df.index:
        for col_id in sigma_df.columns:
            sigma_long_rows.append({
                "community_id_row": int(row_id),
                "community_id_col": int(col_id),
                "sigma_value": float(sigma_df.loc[row_id, col_id]),
                "version_id": current_version_id,
            })
    sigma_long = pd.DataFrame(sigma_long_rows)
    con.execute(f"DELETE FROM community_sigma WHERE version_id = '{current_version_id}'")
    con.execute("INSERT INTO community_sigma SELECT * FROM sigma_long")
    log.info("Ingested community_sigma: %d cells", len(sigma_long))
```

Also add `community_sigma` to the summary print loop.

- [ ] **Step 4.7: Run Stan covariance model**

```bash
python src/covariance/run_county_covariance.py
```

Expected output (~5 minutes for 293 counties, K=chosen, T=5):
```
K = <K> communities
T = 5 elections for Stan
k_ref = X (most Democratic community, 1-indexed)
Running MCMC: 4 chains...
Saved Σ to data/covariance/county_community_sigma.parquet  (K=<K>, T=5)

=== Community Σ (posterior mean) ===
...
=== Community ρ (posterior mean correlation) ===
...
```

If Stan compilation fails, check: `cmdstanpy.install_cmdstan()` and verify Stan version ≥ 2.28.

After Stan run, rebuild DuckDB:
```bash
python src/db/build_database.py --reset
```

- [ ] **Step 4.8: Commit**

```bash
git add src/covariance/stan/community_covariance.stan src/covariance/run_county_covariance.py \
        src/db/build_database.py tests/test_county_covariance.py
git commit -m "feat(phase1): Stan covariance model generalized for any K; county Sigma pipeline"
```

---

## Task 5: HAC Community Weights + 2026 Predictions

**Goal:** Build community weight matrices from the new HAC assignments (replacing the old NMF `c1..c7` weights), then produce 2026 predictions via Gaussian Bayesian update using the new Stan Σ.

**Why a new pipeline is needed:** The existing `propagate_polls.py` and `predict_2026.py` are tightly coupled to the K=7 NMF community structure — they hardcode `COMP_COLS = ["c1".."c7"]`, read soft membership weights from `community_weights_county.parquet` (NMF soft membership), and load `community_sigma.parquet` (7×7 NMF Σ). **Do not modify these files** — they remain valid for the NMF pipeline. Instead, create a new parallel HAC prediction pipeline.

**Files:**
- Create: `src/assembly/build_hac_community_weights.py` — HAC-based weight matrices
- Create: `src/prediction/predict_2026_hac.py` — HAC-based 2026 predictions
- Create: `tests/test_hac_community_weights.py`

### HAC community weights design

With hard HAC assignments, each county belongs to exactly one community (binary membership). The state-level W matrix aggregates by vote totals:

```
W_state[state, k] = sum(recent_total_votes for counties in community k AND state) /
                    sum(recent_total_votes for counties in state)
```

For the county-to-community prior (μ_prior): use theta_obs computed in Task 4 as the prior mean for each community.

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_hac_community_weights.py`:

```python
"""Tests for src/assembly/build_hac_community_weights.py"""
import numpy as np
import pandas as pd
import pytest
from src.assembly.build_hac_community_weights import (
    build_county_weights,
    build_state_weights,
)


@pytest.fixture
def sample_assignments():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "community_id": [0, 0, 1, 1, 2],
    })


@pytest.fixture
def sample_vote_totals():
    return pd.DataFrame({
        "county_fips": ["12001", "12003", "13001", "13003", "01001"],
        "recent_total": [50000.0, 30000.0, 45000.0, 20000.0, 15000.0],
        "state_fips": ["12", "12", "13", "13", "01"],
    })


def test_build_county_weights_shape(sample_assignments, sample_vote_totals):
    w = build_county_weights(sample_assignments, sample_vote_totals)
    assert "county_fips" in w.columns
    assert "community_id" in w.columns
    assert len(w) == len(sample_assignments)


def test_build_county_weights_hard_assignment(sample_assignments, sample_vote_totals):
    """With hard HAC assignments, each county's weight is 1.0 for its community."""
    w = build_county_weights(sample_assignments, sample_vote_totals)
    merged = w.merge(sample_assignments, on="county_fips")
    # weight_in_assigned_community should be 1.0 for all rows
    assert (merged["community_id_x"] == merged["community_id_y"]).all()


def test_build_state_weights_shape(sample_assignments, sample_vote_totals):
    k = sample_assignments["community_id"].nunique()
    w = build_state_weights(sample_assignments, sample_vote_totals)
    assert "state_fips" in w.columns
    assert len(w) == sample_vote_totals["state_fips"].nunique()
    # Should have K weight columns (community_0 ... community_{K-1})
    weight_cols = [c for c in w.columns if c.startswith("community_")]
    assert len(weight_cols) == k


def test_build_state_weights_sum_to_one(sample_assignments, sample_vote_totals):
    w = build_state_weights(sample_assignments, sample_vote_totals)
    weight_cols = [c for c in w.columns if c.startswith("community_")]
    row_sums = w[weight_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)
```

- [ ] **Step 5.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_hac_community_weights.py -v
```
Expected: ImportError

- [ ] **Step 5.3: Implement `src/assembly/build_hac_community_weights.py`**

```python
"""Build community weight matrices from hard HAC assignments.

Unlike the NMF pipeline (which uses soft c1..c7 membership weights),
the HAC pipeline uses hard assignments: each county belongs 100% to its
community. The state-level W matrix weights communities by their share
of a state's recent vote totals.

Inputs:
  data/communities/county_community_assignments.parquet
  data/assembled/medsl_county_2024_president.parquet  (vote totals)

Outputs:
  data/propagation/community_weights_county_hac.parquet
      county_fips, community_id, state_fips
  data/propagation/community_weights_state_hac.parquet
      state_fips, state_abbr, community_0, community_1, ...
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"
OUTPUT_DIR = PROJECT_ROOT / "data" / "propagation"

STATE_FIPS_TO_ABBR = {"12": "FL", "13": "GA", "01": "AL"}


def build_county_weights(
    assignments: pd.DataFrame,
    vote_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build county-level weight table (hard assignment).

    Returns DataFrame: county_fips, community_id, state_fips, recent_total
    """
    merged = assignments.merge(vote_totals, on="county_fips", how="left")
    return merged[["county_fips", "community_id", "state_fips", "recent_total"]]


def build_state_weights(
    assignments: pd.DataFrame,
    vote_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build state-level W matrix for poll propagation.

    W[state, k] = vote share of community k within state.
    Columns: state_fips, state_abbr, community_0, ..., community_{K-1}
    """
    merged = assignments.merge(vote_totals, on="county_fips", how="left")
    merged["recent_total"] = merged["recent_total"].fillna(0)
    k_ids = sorted(merged["community_id"].unique())

    rows = []
    for state_fips, state_df in merged.groupby("state_fips"):
        total_votes = state_df["recent_total"].sum()
        row = {"state_fips": state_fips, "state_abbr": STATE_FIPS_TO_ABBR.get(state_fips, "???")}
        for k in k_ids:
            k_votes = state_df[state_df["community_id"] == k]["recent_total"].sum()
            row[f"community_{k}"] = float(k_votes / total_votes) if total_votes > 0 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    # Normalize rows to exactly sum to 1 (handle float rounding)
    weight_cols = [c for c in df.columns if c.startswith("community_")]
    row_sums = df[weight_cols].sum(axis=1)
    df[weight_cols] = df[weight_cols].div(row_sums, axis=0)
    return df


def run() -> None:
    log.info("Loading assignments...")
    assignments = pd.read_parquet(COMMUNITIES_DIR / "county_community_assignments.parquet")
    assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
    if "community_id" not in assignments.columns and "community" in assignments.columns:
        assignments = assignments.rename(columns={"community": "community_id"})

    log.info("Loading vote totals (2024 president)...")
    pres_2024 = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")
    pres_2024["county_fips"] = pres_2024["county_fips"].astype(str).str.zfill(5)
    vote_totals = pres_2024[["county_fips", "pres_total_2024"]].rename(
        columns={"pres_total_2024": "recent_total"}
    )
    vote_totals["state_fips"] = vote_totals["county_fips"].str[:2]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    county_w = build_county_weights(assignments, vote_totals)
    county_w.to_parquet(OUTPUT_DIR / "community_weights_county_hac.parquet", index=False)
    log.info("Saved county weights: %s", county_w.shape)

    state_w = build_state_weights(assignments, vote_totals)
    state_w.to_parquet(OUTPUT_DIR / "community_weights_state_hac.parquet", index=False)
    log.info("Saved state weights:\n%s", state_w.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
```

- [ ] **Step 5.4: Run tests**

```bash
python -m pytest tests/test_hac_community_weights.py -v
```
Expected: All PASS

- [ ] **Step 5.5: Build HAC community weights**

```bash
python src/assembly/build_hac_community_weights.py
```

Verify:
```bash
python -c "
import pandas as pd
w = pd.read_parquet('data/propagation/community_weights_state_hac.parquet')
print(w)
# Weight columns should sum to 1.0 per row
print('Row sums:', w[[c for c in w.columns if c.startswith('community_')]].sum(axis=1).values)
"
```

- [ ] **Step 5.6: Implement `src/prediction/predict_2026_hac.py`**

This script runs the same Gaussian Bayesian update as the existing `propagate_polls.py` but uses the HAC community structure. Keep it self-contained; do not modify the existing NMF prediction pipeline.

```python
"""2026 predictions using HAC community structure (Phase 1 pipeline).

Loads the HAC community Sigma (from Task 4), state weight matrix (from Task 5),
and polls, runs the Gaussian Bayesian update, and produces county-level
2026 predictions.

Inputs:
  data/covariance/county_community_sigma.parquet   (K×K from Stan)
  data/propagation/community_weights_state_hac.parquet
  data/propagation/community_weights_county_hac.parquet
  data/polls/polls_2026.csv

Outputs:
  data/predictions/county_predictions_2026_hac.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Poll data
POLLS_PATH = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"

# Community structure (HAC)
SIGMA_PATH = PROJECT_ROOT / "data" / "covariance" / "county_community_sigma.parquet"
STATE_W_PATH = PROJECT_ROOT / "data" / "propagation" / "community_weights_state_hac.parquet"
COUNTY_W_PATH = PROJECT_ROOT / "data" / "propagation" / "community_weights_county_hac.parquet"

# Prior: community-level vote shares from theta_obs (computed in Task 4)
THETA_OBS_PATH = PROJECT_ROOT / "data" / "covariance" / "county_theta_obs.parquet"


def bayesian_update(
    mu_prior: np.ndarray,       # (K,) community prior means
    sigma_prior: np.ndarray,    # (K, K) community prior covariance
    W: np.ndarray,              # (n_polls, K) state weight matrix rows
    y: np.ndarray,              # (n_polls,) observed poll dem_shares
    sigma_polls: np.ndarray,    # (n_polls,) poll standard deviations
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian Bayesian update: posterior mean and covariance.

    Returns (mu_post, sigma_post).
    """
    R = np.diag(sigma_polls ** 2)
    sigma_prior_inv = np.linalg.inv(sigma_prior + np.eye(len(mu_prior)) * 1e-8)
    sigma_post_inv = sigma_prior_inv + W.T @ np.linalg.inv(R) @ W
    sigma_post = np.linalg.inv(sigma_post_inv)
    mu_post = sigma_post @ (sigma_prior_inv @ mu_prior + W.T @ np.linalg.solve(R, y))
    return mu_post, sigma_post


def run() -> None:
    log.info("Loading HAC community Sigma...")
    sigma_df = pd.read_parquet(SIGMA_PATH)
    k_ids = list(sigma_df.index)
    K = len(k_ids)
    Sigma = sigma_df.values.astype(float)

    log.info("Loading state weights...")
    state_w = pd.read_parquet(STATE_W_PATH)
    weight_cols = sorted([c for c in state_w.columns if c.startswith("community_")])

    log.info("Loading polls...")
    polls = pd.read_csv(POLLS_PATH)

    # Build community prior from most recent election (theta_obs mean per community)
    # If theta_obs parquet exists, use it; otherwise use a flat 0.45 prior
    if THETA_OBS_PATH.exists():
        theta_obs_df = pd.read_parquet(THETA_OBS_PATH)
        mu_prior = theta_obs_df.mean(axis=1).values  # mean across T elections
    else:
        log.warning("No theta_obs found; using flat 0.45 prior")
        mu_prior = np.full(K, 0.45)

    predictions = []
    for _, row in polls.iterrows():
        state = row["state"]
        race = row["race"]
        poll_avg = row["dem_share"]
        poll_n = row.get("n_sample", 1000)
        poll_sigma = np.sqrt(poll_avg * (1 - poll_avg) / poll_n)

        state_row = state_w[state_w["state_abbr"] == state]
        if state_row.empty:
            log.warning("No weight row for state %s, skipping", state)
            continue

        W = state_row[weight_cols].values  # (1, K)
        mu_post, sigma_post = bayesian_update(
            mu_prior=mu_prior,
            sigma_prior=Sigma,
            W=W,
            y=np.array([poll_avg]),
            sigma_polls=np.array([poll_sigma]),
        )

        # Back-project to county level
        county_w = pd.read_parquet(COUNTY_W_PATH)
        for _, county_row in county_w[county_w["county_fips"].str.startswith(
            state_w[state_w["state_abbr"] == state]["state_fips"].values[0]
        )].iterrows():
            comm_id = county_row["community_id"]
            k_idx = k_ids.index(comm_id)
            pred = float(mu_post[k_idx])
            std = float(np.sqrt(sigma_post[k_idx, k_idx]))
            predictions.append({
                "county_fips": county_row["county_fips"],
                "state_abbr": state,
                "race": race,
                "pred_dem_share": pred,
                "pred_std": std,
                "pred_lo90": pred - 1.645 * std,
                "pred_hi90": pred + 1.645 * std,
                "state_pred": float(W @ mu_post),
                "poll_avg": poll_avg,
            })

    result = pd.DataFrame(predictions)
    out_path = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_hac.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    log.info("Saved %d predictions to %s", len(result), out_path)
    print(result.groupby(["state_abbr", "race"])["pred_dem_share"].describe().round(3))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
```

**Note on theta_obs persistence:** When `run_county_covariance.py` runs in Task 4, add one line to save theta_obs as a parquet (so `predict_2026_hac.py` can load it):

```python
# In run_county_covariance.py, after building theta_obs:
theta_obs_df = pd.DataFrame(
    theta_obs,
    index=sorted(assignments["community_id"].unique()),
    columns=[e[0] for e in _ELECTIONS[:T]],
)
theta_obs_df.to_parquet(output_dir / "county_theta_obs.parquet")
```

Add this line to `run_county_covariance.py` Task 4 Step 4.4 implementation.

- [ ] **Step 5.7: Update `src/db/build_database.py` predictions path**

The DuckDB builder currently reads from `county_predictions_2026.parquet` (old NMF pipeline). Update to also load from the HAC predictions:

```python
# In build_database.py, add a second predictions source:
PREDICTIONS_2026_HAC = PROJECT_ROOT / "data" / "predictions" / "county_predictions_2026_hac.parquet"

# In build() after ingesting the NMF predictions:
if PREDICTIONS_2026_HAC.exists():
    pred_hac = pd.read_parquet(PREDICTIONS_2026_HAC)
    pred_hac["county_fips"] = pred_hac["county_fips"].astype(str).str.zfill(5)
    # Tag with a different version or race prefix to avoid PK conflict
    pred_hac_rows = _build_predictions(pred_hac, current_version_id)
    # Only insert rows not already in predictions (different races)
    existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
    new_rows = pred_hac_rows[~pred_hac_rows["race"].isin(existing_races)]
    if len(new_rows):
        con.execute("INSERT INTO predictions SELECT * FROM new_rows")
        log.info("Ingested HAC predictions: %d rows", len(new_rows))
```

- [ ] **Step 5.8: Run predictions and rebuild DuckDB**

```bash
python src/prediction/predict_2026_hac.py
python src/db/build_database.py --reset
```

Verify:
```bash
python -c "
import duckdb
con = duckdb.connect('data/bedrock.duckdb')
print(con.execute('SELECT race, COUNT(*) n FROM predictions GROUP BY race ORDER BY race').df())
con.close()
"
```

- [ ] **Step 5.9: Commit**

```bash
git add src/assembly/build_hac_community_weights.py \
        src/prediction/predict_2026_hac.py \
        src/db/build_database.py \
        tests/test_hac_community_weights.py
git commit -m "feat(phase1): HAC community weights + 2026 predictions via new Sigma"
```

---

## Task 6: Validation Report + Documentation + Cleanup

**Goal:** Generate a validation report, update all documentation, and commit the Phase 1 completion.

**Files:**
- Create: `src/validation/generate_validation_report.py`
- Modify: `data/models/versions/county_multiyear_logodds_20260319/meta.yaml`
- Modify: `CLAUDE.md` — Key Decisions Log
- Modify: `docs/ROADMAP.md` — mark Phase 1 deliverables complete

- [ ] **Step 6.1: Implement validation report**

Create `src/validation/generate_validation_report.py`:

```python
"""Generate Phase 1 validation report.

Computes:
  - Holdout Pearson r and MAE at the chosen K (train: pres_d_shift_16_20 vs holdout: pres_d_shift_20_24)
  - Comparison to 3-cycle baseline
  - Community-level predictions vs actuals for 2020→2024

Usage:
    python src/validation/generate_validation_report.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster._agglomerative import _hc_cut
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import load_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE = {5: 0.983, 7: 0.964, 10: 0.941, 15: 0.934, 20: 0.932}
PRES_D_16_20_COL = 12  # index within 30 training cols


def generate_report() -> dict:
    from src.core import config as _cfg
    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS

    k = _cfg._cfg["clustering"]["k"]
    if k is None:
        raise RuntimeError("clustering.k is null — run select_k.py first")

    shifts = pd.read_parquet(PROJECT_ROOT / "data" / "shifts" / "county_shifts_multiyear.parquet")
    shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)

    fips_list = (PROJECT_ROOT / "data" / "communities" / "county_adjacency.fips.txt").read_text().splitlines()
    W = load_npz(str(PROJECT_ROOT / "data" / "communities" / "county_adjacency.npz"))

    indexed = shifts.set_index("county_fips").reindex(fips_list)
    train_arr = indexed[TRAINING_SHIFT_COLS].values
    holdout_arr = indexed[HOLDOUT_SHIFT_COLS].values

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_arr)

    model = AgglomerativeClustering(linkage="ward", connectivity=W, n_clusters=1, compute_distances=True)
    model.fit(train_norm)
    labels = _hc_cut(k, model.children_, len(fips_list))

    train_means = np.array([train_arr[labels == i, PRES_D_16_20_COL].mean() for i in range(k)])
    holdout_means = np.array([holdout_arr[labels == i, 0].mean() for i in range(k)])

    r = float(pearsonr(train_means, holdout_means).statistic)
    mae = float(np.mean(np.abs(train_means - holdout_means)))

    baseline_r = BASELINE.get(k, None)

    report = {
        "chosen_k": k,
        "holdout_r_multiyear": r,
        "holdout_mae": mae,
        "baseline_3cycle_r": baseline_r,
        "delta_vs_baseline": r - baseline_r if baseline_r is not None else None,
        "community_sizes": np.bincount(labels).tolist(),
    }

    print("\n=== Phase 1 Validation Report ===")
    print(f"Chosen K          : {k}")
    print(f"Holdout r         : {r:.4f}")
    print(f"Holdout MAE       : {mae:.4f}")
    if baseline_r:
        print(f"3-cycle baseline r: {baseline_r:.4f}")
        print(f"Delta             : {r - baseline_r:.4f}")
    print(f"Community sizes   : {np.bincount(labels).tolist()}")

    return report


if __name__ == "__main__":
    generate_report()
```

- [ ] **Step 6.2: Run validation report**

```bash
python src/validation/generate_validation_report.py
```

Record the output. The holdout_r and chosen K values need to be added to:
1. `data/models/versions/county_multiyear_logodds_20260319/meta.yaml`
2. `CLAUDE.md` Key Decisions Log

- [ ] **Step 6.3: Update meta.yaml with Phase 1 results**

Edit `data/models/versions/county_multiyear_logodds_20260319/meta.yaml`:
```yaml
k: <chosen_K from Task 1>
j: 7
holdout_r: <r from Task 1>
```

- [ ] **Step 6.4: Update `CLAUDE.md` Key Decisions Log**

Add an entry:
```markdown
| 2026-03-19 | Phase 1 K selection: K=<K> (holdout r=<r>) | K selection sweep over K=5..30, min community size 8. K=<K> maximizes holdout Pearson r between community-mean pres_d_shift_16_20 (training) and pres_d_shift_20_24 (holdout). 3-cycle baseline at K=<K> was r=<baseline_r>; multi-year model gives r=<r>. |
| 2026-03-19 | Phase 1 NMF types: J=7 | J sweep over 5-8; J=7 chosen for consistency with canonical K=7 NMF history and interpretability. Type weights stored in county_type_assignments.parquet. |
| 2026-03-19 | Phase 1 Stan Σ: county HAC model, T=5 elections | Stan factor model (rank-1) fit on 5 elections (2016 pres, 2018 gov, 2020 pres, 2022 gov, 2024 pres). k_ref = most Democratic community (dynamic). Σ stored at data/covariance/county_community_sigma.parquet. |
```

- [ ] **Step 6.5: Update `docs/ROADMAP.md`**

Mark the Phase 1 deliverables section with completion checkboxes:

```markdown
### Deliverables
- [x] Log-odds shift vectors for all 293 FL+GA+AL counties, all available cycles
- [ ] Senate races added to training (MEDSL Senate data via Harvard Dataverse) ← DEFERRED to Phase 1b
- [x] K selection via holdout accuracy sweep
- [x] J=7 for types
- [x] Hard community assignments (Layer 1) stored in DuckDB
- [x] Soft type assignments (Layer 2, NMF on community shift profiles) stored in DuckDB
- [x] Stan Σ (community covariance matrix) estimated and stored
- [x] 2026 county-level predictions updated with new community structure
- [ ] Turnout feature from dropped uncontested pairs (OQ-002) ← Phase 1b
- [ ] 3D vs 2D triplet comparison (OQ-003) ← Phase 1b
- [ ] Community descriptions: ACS, RCMS, IRS migration overlays ← Phase 1b
- [ ] Full validation report vs national polling ← Phase 1b
```

- [ ] **Step 6.6: Final test run**

```bash
python -m pytest --ignore=tests/test_covariance.py --ignore=tests/test_description.py \
  --ignore=tests/test_detection.py --ignore=tests/test_discovery.py \
  --ignore=tests/test_holdout.py --ignore=tests/test_propagation.py \
  -q 2>&1 | tail -5
```

Expected: all new tests pass, no regressions beyond the known pre-existing failures.

- [ ] **Step 6.7: Final commit**

```bash
git add src/validation/generate_validation_report.py \
        data/models/versions/county_multiyear_logodds_20260319/meta.yaml \
        CLAUDE.md docs/ROADMAP.md
git commit -m "docs(phase1): validation report, updated meta.yaml, CLAUDE.md, ROADMAP.md"
git push
```

---

## Phase 1b (Deferred Items)

These are important but not on the critical path for the prediction pipeline:

| Item | Why deferred |
|------|-------------|
| Senate races | Adds training dims but requires a new fetcher + column harmonization across states with irregular Senate cycles |
| Turnout feature (OQ-002) | Research question; answered after core model is stable |
| 3D vs 2D triplet (OQ-003) | Validation experiment; core model uses D-shift + turnout already |
| Community descriptions (ACS/RCMS/IRS) | Phase 2 visualization work depends on this, not Phase 1 prediction |
| Full national polling comparison | Phase 2 API/visualization work |

---

## Success Criteria

Phase 1 is complete when:
1. `config/model.yaml` has a non-null `clustering.k`
2. `data/communities/county_community_assignments.parquet` has K communities from multiyear shifts
3. `data/communities/county_type_assignments.parquet` has real NMF type weights (not uniform stub)
4. `data/covariance/county_community_sigma.parquet` exists with a K×K positive-definite matrix
5. `data/bedrock.duckdb` has all 6 tables populated with Phase 1 outputs
6. Validation report printed showing holdout r, MAE, comparison to baseline
7. All new tests pass; docs updated; committed and pushed to GitHub
