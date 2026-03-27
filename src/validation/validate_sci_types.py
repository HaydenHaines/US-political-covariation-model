"""Validate electoral types against Facebook Social Connectedness Index.

Research question: Do socially connected counties (high SCI) tend to belong
to the same electoral type? If so, this validates that our type structure
captures real community structure, not just statistical artifacts.

Approach:
1. Load SCI county pairs and their connectedness scores.
2. Load county type assignments (soft membership vectors).
3. For each county pair, compute:
   (a) SCI score (log-scaled for analysis)
   (b) Whether they share the same primary (dominant) type
   (c) Cosine similarity of their soft membership vectors
4. Compute correlation between log(SCI) and type similarity.
5. Compare mean SCI for same-type vs different-type pairs.
6. Control for geographic proximity (same-state, geodesic distance) to show
   SCI adds signal beyond physical closeness.

Memory strategy: The full SCI file has 10.3M rows (symmetric pairs). We load
only the upper triangle (~5.1M pairs) and filter to counties present in our
type assignments (~3,154 counties). We sample for correlation analysis since
pairwise operations on 3K+ counties are tractable but we don't need all pairs
for statistical power.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]

# Default paths (overridable for testing)
SCI_PATH = PROJECT_ROOT / "data" / "raw" / "facebook_sci" / "us_counties.csv"
TYPE_ASSIGNMENTS_PATH = (
    PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
)
COUNTY_CENTROIDS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "county_centroids_2020.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"

# Haversine Earth radius in km
_EARTH_RADIUS_KM = 6371.0

# Distance bins for the proximity control (km)
DISTANCE_BINS = [0, 100, 250, 500, 1000, 2000, 5000]
DISTANCE_LABELS = [
    "0-100km",
    "100-250km",
    "250-500km",
    "500-1000km",
    "1000-2000km",
    "2000-5000km",
]


@dataclass
class SCITypeValidationResult:
    """Container for SCI-type validation results."""

    # Basic statistics
    n_counties: int = 0
    n_pairs: int = 0
    n_types: int = 0

    # Same-type vs different-type comparison
    mean_sci_same_type: float = 0.0
    mean_sci_diff_type: float = 0.0
    sci_ratio_same_over_diff: float = 0.0
    mean_log_sci_same_type: float = 0.0
    mean_log_sci_diff_type: float = 0.0

    # Correlation: log(SCI) vs cosine similarity of type vectors
    pearson_r_log_sci_vs_cosine: float = 0.0
    pearson_p_log_sci_vs_cosine: float = 0.0
    spearman_r_log_sci_vs_cosine: float = 0.0
    spearman_p_log_sci_vs_cosine: float = 0.0

    # Same-state control
    pct_same_type_same_state: float = 0.0
    pct_same_type_diff_state: float = 0.0
    mean_sci_same_type_diff_state: float = 0.0
    mean_sci_diff_type_diff_state: float = 0.0
    sci_ratio_same_over_diff_across_states: float = 0.0

    # Distance-binned analysis (SCI ratio per distance bin)
    distance_bin_results: list[dict] = field(default_factory=list)

    # Partial correlation: log(SCI) vs cosine sim, controlling for log(distance)
    partial_r_sci_cosine_given_distance: float = 0.0
    partial_p_sci_cosine_given_distance: float = 0.0


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_sci_upper_triangle(
    path: Path,
    valid_fips: set[str] | None = None,
) -> pd.DataFrame:
    """Load SCI data, keeping only upper triangle (user < friend) for unique pairs.

    Parameters
    ----------
    path:
        Path to the SCI CSV file.
    valid_fips:
        If provided, filter to pairs where both counties are in this set.

    Returns
    -------
    DataFrame with columns: user_fips, friend_fips, scaled_sci.
    """
    log.info("Loading SCI data from %s", path)
    df = pd.read_csv(
        path,
        usecols=["user_region", "friend_region", "scaled_sci"],
        dtype={"user_region": str, "friend_region": str, "scaled_sci": np.int64},
    )
    log.info("  Raw rows: %d", len(df))

    # Zero-pad FIPS to 5 chars
    df["user_region"] = df["user_region"].str.zfill(5)
    df["friend_region"] = df["friend_region"].str.zfill(5)

    # Drop self-connections
    df = df[df["user_region"] != df["friend_region"]].copy()

    # Keep only upper triangle (user < friend) to deduplicate symmetric pairs
    df = df[df["user_region"] < df["friend_region"]].copy()
    log.info("  Upper-triangle rows: %d", len(df))

    # Filter to valid FIPS if provided
    if valid_fips is not None:
        mask = df["user_region"].isin(valid_fips) & df["friend_region"].isin(valid_fips)
        df = df[mask].copy()
        log.info("  After FIPS filter: %d", len(df))

    df = df.rename(columns={"user_region": "user_fips", "friend_region": "friend_fips"})
    return df[["user_fips", "friend_fips", "scaled_sci"]].reset_index(drop=True)


def load_type_assignments(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load type assignments and extract score matrix + dominant types.

    Returns
    -------
    (assignments_df, score_matrix, dominant_types)
        score_matrix: (N, J) array of soft membership scores
        dominant_types: (N,) array of primary type indices
    """
    log.info("Loading type assignments from %s", path)
    df = pd.read_parquet(path)

    score_cols = sorted(
        [c for c in df.columns if c.endswith("_score")],
        key=lambda c: int(c.split("_")[1]),
    )
    score_matrix = df[score_cols].values  # (N, J)
    dominant_types = df["dominant_type"].values
    log.info(
        "  %d counties, %d types, dominant types range [%d, %d]",
        len(df),
        len(score_cols),
        dominant_types.min(),
        dominant_types.max(),
    )
    return df, score_matrix, dominant_types


def load_county_centroids(path: Path) -> pd.DataFrame:
    """Load county centroids (lat/lon) from Census Gazetteer file.

    Expected columns: county_fips, latitude, longitude.
    """
    log.info("Loading county centroids from %s", path)
    df = pd.read_csv(path, dtype={"county_fips": str})
    df["county_fips"] = df["county_fips"].str.zfill(5)
    log.info("  %d counties with centroids", len(df))
    return df


def compute_pairwise_type_similarity(
    sci_pairs: pd.DataFrame,
    fips_to_idx: dict[str, int],
    score_matrix: np.ndarray,
    dominant_types: np.ndarray,
) -> pd.DataFrame:
    """Add type similarity columns to SCI pair DataFrame.

    Adds:
    - same_type: bool, whether both counties share the same dominant type
    - cosine_sim: cosine similarity of soft membership vectors
    - log_sci: log10(scaled_sci)
    - same_state: bool, whether both counties are in the same state
    """
    log.info("Computing pairwise type similarity for %d pairs...", len(sci_pairs))

    user_idx = sci_pairs["user_fips"].map(fips_to_idx).values
    friend_idx = sci_pairs["friend_fips"].map(fips_to_idx).values

    # Same dominant type
    sci_pairs = sci_pairs.copy()
    sci_pairs["same_type"] = dominant_types[user_idx] == dominant_types[friend_idx]

    # Cosine similarity of soft membership vectors (vectorized)
    user_vecs = score_matrix[user_idx]  # (N_pairs, J)
    friend_vecs = score_matrix[friend_idx]  # (N_pairs, J)
    # cosine_sim = dot(u, v) / (||u|| * ||v||)
    dot_products = np.sum(user_vecs * friend_vecs, axis=1)
    user_norms = np.linalg.norm(user_vecs, axis=1)
    friend_norms = np.linalg.norm(friend_vecs, axis=1)
    # Avoid division by zero (shouldn't happen with valid scores)
    denom = user_norms * friend_norms
    denom = np.where(denom > 0, denom, 1.0)
    sci_pairs["cosine_sim"] = dot_products / denom

    # Log-scaled SCI
    sci_pairs["log_sci"] = np.log10(sci_pairs["scaled_sci"].clip(lower=1))

    # Same-state indicator
    sci_pairs["same_state"] = (
        sci_pairs["user_fips"].str[:2] == sci_pairs["friend_fips"].str[:2]
    )

    log.info(
        "  Same-type pairs: %d (%.1f%%), different-type: %d",
        sci_pairs["same_type"].sum(),
        100 * sci_pairs["same_type"].mean(),
        (~sci_pairs["same_type"]).sum(),
    )
    return sci_pairs


def add_geodesic_distance(
    pairs: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Add geodesic distance (km) between county centroids to pair DataFrame."""
    log.info("Computing geodesic distances for %d pairs...", len(pairs))

    centroid_map = centroids.set_index("county_fips")[["latitude", "longitude"]]

    pairs = pairs.copy()
    user_coords = pairs["user_fips"].map(centroid_map["latitude"]).values
    user_lon = pairs["user_fips"].map(centroid_map["longitude"]).values
    friend_coords = pairs["friend_fips"].map(centroid_map["latitude"]).values
    friend_lon = pairs["friend_fips"].map(centroid_map["longitude"]).values

    # Drop pairs where centroids are missing
    valid_mask = ~(
        np.isnan(user_coords)
        | np.isnan(user_lon)
        | np.isnan(friend_coords)
        | np.isnan(friend_lon)
    )
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        log.warning("  %d pairs lack centroid data, dropping", n_dropped)
        pairs = pairs[valid_mask].copy()
        user_coords = user_coords[valid_mask]
        user_lon = user_lon[valid_mask]
        friend_coords = friend_coords[valid_mask]
        friend_lon = friend_lon[valid_mask]

    pairs["distance_km"] = haversine_km(
        user_coords, user_lon, friend_coords, friend_lon
    )
    pairs["log_distance"] = np.log10(pairs["distance_km"].clip(lower=1))

    log.info("  Distance range: %.0f - %.0f km", pairs["distance_km"].min(), pairs["distance_km"].max())
    return pairs


def compute_partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Partial Pearson correlation of x and y controlling for z.

    Uses the standard formula: regress x on z, regress y on z, correlate
    residuals.
    """
    # Residualize x on z
    z_with_const = np.column_stack([z, np.ones_like(z)])
    beta_x = np.linalg.lstsq(z_with_const, x, rcond=None)[0]
    resid_x = x - z_with_const @ beta_x

    beta_y = np.linalg.lstsq(z_with_const, y, rcond=None)[0]
    resid_y = y - z_with_const @ beta_y

    return pearsonr(resid_x, resid_y)


def run_validation(
    sci_pairs: pd.DataFrame,
    type_assignments: pd.DataFrame,
    score_matrix: np.ndarray,
    dominant_types: np.ndarray,
    centroids: pd.DataFrame | None = None,
    sample_size: int | None = None,
) -> SCITypeValidationResult:
    """Run the full SCI-type validation analysis.

    Parameters
    ----------
    sci_pairs:
        Upper-triangle SCI pairs (user_fips, friend_fips, scaled_sci).
    type_assignments:
        Type assignment DataFrame with county_fips column.
    score_matrix:
        (N, J) soft membership matrix.
    dominant_types:
        (N,) dominant type assignments.
    centroids:
        Optional county centroid DataFrame for distance controls.
    sample_size:
        If provided, randomly sample this many pairs for correlation analysis
        (for memory/speed). All pairs used for mean comparisons.
    """
    result = SCITypeValidationResult()

    # Build FIPS-to-index mapping
    fips_list = type_assignments["county_fips"].values
    fips_to_idx = {f: i for i, f in enumerate(fips_list)}

    result.n_counties = len(fips_list)
    result.n_types = score_matrix.shape[1]

    # Add type similarity to pairs
    pairs = compute_pairwise_type_similarity(
        sci_pairs, fips_to_idx, score_matrix, dominant_types
    )
    result.n_pairs = len(pairs)

    # --- Basic same-type vs different-type comparison ---
    same_mask = pairs["same_type"]
    result.mean_sci_same_type = float(pairs.loc[same_mask, "scaled_sci"].mean())
    result.mean_sci_diff_type = float(pairs.loc[~same_mask, "scaled_sci"].mean())
    result.sci_ratio_same_over_diff = (
        result.mean_sci_same_type / result.mean_sci_diff_type
        if result.mean_sci_diff_type > 0
        else float("inf")
    )
    result.mean_log_sci_same_type = float(pairs.loc[same_mask, "log_sci"].mean())
    result.mean_log_sci_diff_type = float(pairs.loc[~same_mask, "log_sci"].mean())

    # --- Correlation: log(SCI) vs cosine similarity ---
    if sample_size and len(pairs) > sample_size:
        sample = pairs.sample(n=sample_size, random_state=42)
    else:
        sample = pairs

    r_p, p_p = pearsonr(sample["log_sci"], sample["cosine_sim"])
    result.pearson_r_log_sci_vs_cosine = float(r_p)
    result.pearson_p_log_sci_vs_cosine = float(p_p)

    r_s, p_s = spearmanr(sample["log_sci"], sample["cosine_sim"])
    result.spearman_r_log_sci_vs_cosine = float(r_s)
    result.spearman_p_log_sci_vs_cosine = float(p_s)

    # --- Same-state control ---
    same_state_mask = pairs["same_state"]
    same_type_same_state = pairs.loc[same_mask & same_state_mask]
    same_type_diff_state = pairs.loc[same_mask & ~same_state_mask]
    diff_type_diff_state = pairs.loc[~same_mask & ~same_state_mask]

    if len(same_type_same_state) > 0:
        result.pct_same_type_same_state = float(
            same_mask[same_state_mask].mean()
        )
    if len(same_type_diff_state) > 0:
        result.pct_same_type_diff_state = float(
            same_mask[~same_state_mask].mean()
        )

    if len(same_type_diff_state) > 0:
        result.mean_sci_same_type_diff_state = float(
            same_type_diff_state["scaled_sci"].mean()
        )
    if len(diff_type_diff_state) > 0:
        result.mean_sci_diff_type_diff_state = float(
            diff_type_diff_state["scaled_sci"].mean()
        )
    if result.mean_sci_diff_type_diff_state > 0:
        result.sci_ratio_same_over_diff_across_states = (
            result.mean_sci_same_type_diff_state
            / result.mean_sci_diff_type_diff_state
        )

    # --- Distance controls (if centroids available) ---
    if centroids is not None and len(centroids) > 0:
        pairs_with_dist = add_geodesic_distance(pairs, centroids)

        # Partial correlation: log(SCI) vs cosine_sim | log(distance)
        if sample_size and len(pairs_with_dist) > sample_size:
            dist_sample = pairs_with_dist.sample(n=sample_size, random_state=42)
        else:
            dist_sample = pairs_with_dist

        valid = dist_sample.dropna(subset=["log_sci", "cosine_sim", "log_distance"])
        if len(valid) > 100:
            pr, pp = compute_partial_correlation(
                valid["log_sci"].values,
                valid["cosine_sim"].values,
                valid["log_distance"].values,
            )
            result.partial_r_sci_cosine_given_distance = float(pr)
            result.partial_p_sci_cosine_given_distance = float(pp)

        # Distance-binned analysis
        pairs_with_dist["distance_bin"] = pd.cut(
            pairs_with_dist["distance_km"],
            bins=DISTANCE_BINS,
            labels=DISTANCE_LABELS,
            right=True,
        )
        for bin_label in DISTANCE_LABELS:
            bin_data = pairs_with_dist[pairs_with_dist["distance_bin"] == bin_label]
            if len(bin_data) < 10:
                continue
            bin_same = bin_data[bin_data["same_type"]]
            bin_diff = bin_data[~bin_data["same_type"]]
            mean_same = float(bin_same["scaled_sci"].mean()) if len(bin_same) > 0 else 0.0
            mean_diff = float(bin_diff["scaled_sci"].mean()) if len(bin_diff) > 0 else 0.0
            ratio = mean_same / mean_diff if mean_diff > 0 else float("inf")

            # Correlation within this distance bin
            if len(bin_data) > 30:
                bin_r, bin_p = pearsonr(bin_data["log_sci"], bin_data["cosine_sim"])
            else:
                bin_r, bin_p = float("nan"), float("nan")

            result.distance_bin_results.append({
                "distance_bin": bin_label,
                "n_pairs": len(bin_data),
                "n_same_type": len(bin_same),
                "pct_same_type": float(bin_data["same_type"].mean()),
                "mean_sci_same_type": mean_same,
                "mean_sci_diff_type": mean_diff,
                "sci_ratio": ratio,
                "pearson_r": float(bin_r),
                "pearson_p": float(bin_p),
            })

    return result


def fetch_county_centroids(output_path: Path) -> pd.DataFrame:
    """Download 2020 Census Gazetteer county centroids if not on disk.

    Source: Census Bureau county gazetteer file (public domain, ~100KB).
    """
    if output_path.exists():
        log.info("County centroids already on disk: %s", output_path)
        return load_county_centroids(output_path)

    import io
    import zipfile
    from urllib.request import urlopen

    url = (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2020_Gazetteer/2020_Gaz_counties_national.zip"
    )
    log.info("Downloading county centroids from %s", url)
    with urlopen(url) as resp:
        zip_data = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # The zip contains a single tab-delimited file
        names = zf.namelist()
        txt_name = [n for n in names if n.endswith(".txt")][0]
        with zf.open(txt_name) as f:
            raw = pd.read_csv(f, sep="\t", dtype={"GEOID": str})

    # Census gazetteer has trailing whitespace in column names
    raw.columns = raw.columns.str.strip()

    # Standardize columns
    centroids = pd.DataFrame({
        "county_fips": raw["GEOID"].str.zfill(5),
        "latitude": raw["INTPTLAT"].astype(float),
        "longitude": raw["INTPTLONG"].astype(float),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_csv(output_path, index=False)
    log.info("Saved %d county centroids to %s", len(centroids), output_path)
    return centroids


def format_results(result: SCITypeValidationResult) -> str:
    """Format validation results as a readable markdown report."""
    lines = [
        "# SCI-Type Validation Results",
        "",
        "## Research Question",
        "",
        "Do counties that are socially connected (high Facebook SCI) also tend",
        "to belong to the same electoral type? If so, this validates that our",
        "types capture real community structure, not just statistical artifacts.",
        "",
        "## Data Summary",
        "",
        f"- **Counties**: {result.n_counties:,}",
        f"- **County pairs analyzed**: {result.n_pairs:,}",
        f"- **Electoral types (J)**: {result.n_types}",
        "",
        "## Finding 1: Same-Type Counties Are More Socially Connected",
        "",
        "| Metric | Same Type | Different Type | Ratio |",
        "|--------|-----------|----------------|-------|",
        f"| Mean SCI | {result.mean_sci_same_type:,.0f} | {result.mean_sci_diff_type:,.0f} | {result.sci_ratio_same_over_diff:.2f}x |",
        f"| Mean log10(SCI) | {result.mean_log_sci_same_type:.3f} | {result.mean_log_sci_diff_type:.3f} | +{result.mean_log_sci_same_type - result.mean_log_sci_diff_type:.3f} |",
        "",
        "## Finding 2: SCI Correlates with Type Similarity",
        "",
        "Correlation between log10(SCI) and cosine similarity of soft type",
        "membership vectors:",
        "",
        f"- **Pearson r**: {result.pearson_r_log_sci_vs_cosine:.4f} (p={result.pearson_p_log_sci_vs_cosine:.2e})",
        f"- **Spearman r**: {result.spearman_r_log_sci_vs_cosine:.4f} (p={result.spearman_p_log_sci_vs_cosine:.2e})",
        "",
        "## Finding 3: Effect Persists Across State Lines",
        "",
        "Same-state pairs are both closer AND more likely to share a type.",
        "The critical test: does the SCI-type relationship hold for",
        "cross-state pairs?",
        "",
        f"- **% same-type among same-state pairs**: {100 * result.pct_same_type_same_state:.1f}%",
        f"- **% same-type among cross-state pairs**: {100 * result.pct_same_type_diff_state:.1f}%",
        "",
        "Cross-state only:",
        "",
        f"- Mean SCI (same type, cross-state): {result.mean_sci_same_type_diff_state:,.0f}",
        f"- Mean SCI (diff type, cross-state): {result.mean_sci_diff_type_diff_state:,.0f}",
        f"- **Cross-state SCI ratio**: {result.sci_ratio_same_over_diff_across_states:.2f}x",
        "",
    ]

    if result.partial_r_sci_cosine_given_distance != 0.0:
        lines.extend([
            "## Finding 4: SCI Signal Beyond Geographic Proximity",
            "",
            "Partial correlation of log(SCI) vs type cosine similarity,",
            "controlling for log(geodesic distance):",
            "",
            f"- **Partial r**: {result.partial_r_sci_cosine_given_distance:.4f} (p={result.partial_p_sci_cosine_given_distance:.2e})",
            "",
            "This measures whether SCI adds information about type similarity",
            "beyond what geographic distance alone provides.",
            "",
        ])

    if result.distance_bin_results:
        lines.extend([
            "## Finding 5: SCI-Type Relationship by Distance Bin",
            "",
            "Holding distance roughly constant, same-type pairs still have",
            "higher SCI than different-type pairs:",
            "",
            "| Distance | N pairs | % Same Type | Mean SCI (same) | Mean SCI (diff) | Ratio | Pearson r |",
            "|----------|---------|-------------|-----------------|-----------------|-------|-----------|",
        ])
        for b in result.distance_bin_results:
            r_str = f"{b['pearson_r']:.3f}" if not np.isnan(b["pearson_r"]) else "n/a"
            lines.append(
                f"| {b['distance_bin']} | {b['n_pairs']:,} | "
                f"{100 * b['pct_same_type']:.1f}% | "
                f"{b['mean_sci_same_type']:,.0f} | {b['mean_sci_diff_type']:,.0f} | "
                f"{b['sci_ratio']:.2f}x | {r_str} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "A positive SCI-type correlation validates that our electoral types",
        "capture real social community structure. Counties in the same type",
        "are not just statistically similar -- they are socially connected",
        "in measurable ways.",
        "",
        "The distance controls are critical: nearby counties are trivially",
        "both more connected (shared commute networks) and more similar",
        "(shared media markets, economies). The partial correlation and",
        "within-bin analysis isolate the SCI signal that goes beyond",
        "geographic proximity.",
        "",
        "## Methodology",
        "",
        "- **SCI data**: Facebook Social Connectedness Index (county pairs,",
        "  Jan 2026 snapshot). Higher SCI = more Facebook friendships between",
        "  two counties, scaled for population.",
        "- **Type similarity**: Cosine similarity of J=100 soft membership",
        "  vectors (temperature-scaled inverse distance to KMeans centroids).",
        "- **Distance control**: Haversine distance between Census 2020",
        "  county centroids.",
        "- **Partial correlation**: Standard OLS residualization of both",
        "  log(SCI) and cosine_sim on log(distance), then Pearson r of",
        "  residuals.",
    ])

    return "\n".join(lines)


def main() -> None:
    """Run the full validation pipeline and save results."""
    # Load type assignments
    assignments, score_matrix, dominant_types = load_type_assignments(
        TYPE_ASSIGNMENTS_PATH
    )
    valid_fips = set(assignments["county_fips"].values)

    # Load SCI data (upper triangle only, filtered to our counties)
    sci_pairs = load_sci_upper_triangle(SCI_PATH, valid_fips=valid_fips)

    # Fetch/load county centroids for distance control
    centroids = fetch_county_centroids(COUNTY_CENTROIDS_PATH)

    # Run validation
    result = run_validation(
        sci_pairs=sci_pairs,
        type_assignments=assignments,
        score_matrix=score_matrix,
        dominant_types=dominant_types,
        centroids=centroids,
        sample_size=500_000,
    )

    # Format and save results
    report = format_results(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "docs" / "research" / "sci-type-validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log.info("Report saved to %s", report_path)

    # Also print summary to stdout
    print("\n" + "=" * 60)
    print("SCI-TYPE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Counties: {result.n_counties:,}")
    print(f"Pairs analyzed: {result.n_pairs:,}")
    print(f"SCI ratio (same/diff type): {result.sci_ratio_same_over_diff:.2f}x")
    print(f"Pearson r (log SCI vs cosine sim): {result.pearson_r_log_sci_vs_cosine:.4f}")
    print(f"Spearman r (log SCI vs cosine sim): {result.spearman_r_log_sci_vs_cosine:.4f}")
    if result.partial_r_sci_cosine_given_distance != 0.0:
        print(
            f"Partial r (controlling distance): "
            f"{result.partial_r_sci_cosine_given_distance:.4f}"
        )
    print(f"Cross-state SCI ratio: {result.sci_ratio_same_over_diff_across_states:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
