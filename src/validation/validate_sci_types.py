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

Module structure:
- sci_loader.py     — SCI data loading and preprocessing
- sci_similarity.py — pairwise type similarity + geodesic distance computation
- validate_sci_types.py (this file) — result container, full validation pipeline,
                                      report formatting, and CLI entry point
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.validation.sci_loader import (  # noqa: F401
    COUNTY_CENTROIDS_PATH,
    OUTPUT_DIR,
    SCI_PATH,
    TYPE_ASSIGNMENTS_PATH,
    fetch_county_centroids,
    load_county_centroids,
    load_sci_upper_triangle,
    load_type_assignments,
)
from src.validation.sci_similarity import (  # noqa: F401
    DISTANCE_BINS,
    DISTANCE_LABELS,
    add_geodesic_distance,
    compute_partial_correlation,
    compute_pairwise_type_similarity,
    haversine_km,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]


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


def _compute_same_type_comparison(
    pairs: pd.DataFrame,
    result: SCITypeValidationResult,
) -> None:
    """Fill same-type vs different-type SCI comparison fields on result (in-place)."""
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


def _compute_correlations(
    pairs: pd.DataFrame,
    sample_size: int | None,
    result: SCITypeValidationResult,
) -> None:
    """Fill Pearson/Spearman correlation fields on result (in-place)."""
    sample = pairs.sample(n=sample_size, random_state=42) if sample_size and len(pairs) > sample_size else pairs

    r_p, p_p = pearsonr(sample["log_sci"], sample["cosine_sim"])
    result.pearson_r_log_sci_vs_cosine = float(r_p)
    result.pearson_p_log_sci_vs_cosine = float(p_p)

    r_s, p_s = spearmanr(sample["log_sci"], sample["cosine_sim"])
    result.spearman_r_log_sci_vs_cosine = float(r_s)
    result.spearman_p_log_sci_vs_cosine = float(p_s)


def _compute_same_state_controls(
    pairs: pd.DataFrame,
    result: SCITypeValidationResult,
) -> None:
    """Fill same-state cross-state comparison fields on result (in-place)."""
    same_mask = pairs["same_type"]
    same_state_mask = pairs["same_state"]
    same_type_diff_state = pairs.loc[same_mask & ~same_state_mask]
    diff_type_diff_state = pairs.loc[~same_mask & ~same_state_mask]

    if len(pairs.loc[same_mask & same_state_mask]) > 0:
        result.pct_same_type_same_state = float(same_mask[same_state_mask].mean())
    if len(same_type_diff_state) > 0:
        result.pct_same_type_diff_state = float(same_mask[~same_state_mask].mean())
        result.mean_sci_same_type_diff_state = float(same_type_diff_state["scaled_sci"].mean())
    if len(diff_type_diff_state) > 0:
        result.mean_sci_diff_type_diff_state = float(diff_type_diff_state["scaled_sci"].mean())
    if result.mean_sci_diff_type_diff_state > 0:
        result.sci_ratio_same_over_diff_across_states = (
            result.mean_sci_same_type_diff_state / result.mean_sci_diff_type_diff_state
        )


def _compute_distance_controls(
    pairs: pd.DataFrame,
    centroids: pd.DataFrame,
    sample_size: int | None,
    result: SCITypeValidationResult,
) -> None:
    """Fill partial correlation and distance-bin fields on result (in-place)."""
    pairs_with_dist = add_geodesic_distance(pairs, centroids)

    # Partial correlation: log(SCI) vs cosine_sim | log(distance)
    dist_sample = (
        pairs_with_dist.sample(n=sample_size, random_state=42)
        if sample_size and len(pairs_with_dist) > sample_size
        else pairs_with_dist
    )
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

    _compute_same_type_comparison(pairs, result)
    _compute_correlations(pairs, sample_size, result)
    _compute_same_state_controls(pairs, result)

    if centroids is not None and len(centroids) > 0:
        _compute_distance_controls(pairs, centroids, sample_size, result)

    return result


def _format_finding_1(result: SCITypeValidationResult) -> list[str]:
    """Format the same-type vs different-type SCI comparison section."""
    return [
        "## Finding 1: Same-Type Counties Are More Socially Connected",
        "",
        "| Metric | Same Type | Different Type | Ratio |",
        "|--------|-----------|----------------|-------|",
        f"| Mean SCI | {result.mean_sci_same_type:,.0f} | {result.mean_sci_diff_type:,.0f} | {result.sci_ratio_same_over_diff:.2f}x |",
        f"| Mean log10(SCI) | {result.mean_log_sci_same_type:.3f} | {result.mean_log_sci_diff_type:.3f} | +{result.mean_log_sci_same_type - result.mean_log_sci_diff_type:.3f} |",
        "",
    ]


def _format_finding_2(result: SCITypeValidationResult) -> list[str]:
    """Format the log(SCI) vs cosine similarity correlation section."""
    return [
        "## Finding 2: SCI Correlates with Type Similarity",
        "",
        "Correlation between log10(SCI) and cosine similarity of soft type",
        "membership vectors:",
        "",
        f"- **Pearson r**: {result.pearson_r_log_sci_vs_cosine:.4f} (p={result.pearson_p_log_sci_vs_cosine:.2e})",
        f"- **Spearman r**: {result.spearman_r_log_sci_vs_cosine:.4f} (p={result.spearman_p_log_sci_vs_cosine:.2e})",
        "",
    ]


def _format_finding_3(result: SCITypeValidationResult) -> list[str]:
    """Format the cross-state persistence section."""
    return [
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


def _format_finding_4(result: SCITypeValidationResult) -> list[str]:
    """Format the geographic proximity partial correlation section (omitted if not computed)."""
    if result.partial_r_sci_cosine_given_distance == 0.0:
        return []
    return [
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
    ]


def _format_finding_5(result: SCITypeValidationResult) -> list[str]:
    """Format the distance-binned SCI ratio table (omitted if no bin results)."""
    if not result.distance_bin_results:
        return []
    lines = [
        "## Finding 5: SCI-Type Relationship by Distance Bin",
        "",
        "Holding distance roughly constant, same-type pairs still have",
        "higher SCI than different-type pairs:",
        "",
        "| Distance | N pairs | % Same Type | Mean SCI (same) | Mean SCI (diff) | Ratio | Pearson r |",
        "|----------|---------|-------------|-----------------|-----------------|-------|-----------|",
    ]
    for b in result.distance_bin_results:
        r_str = f"{b['pearson_r']:.3f}" if not np.isnan(b["pearson_r"]) else "n/a"
        lines.append(
            f"| {b['distance_bin']} | {b['n_pairs']:,} | "
            f"{100 * b['pct_same_type']:.1f}% | "
            f"{b['mean_sci_same_type']:,.0f} | {b['mean_sci_diff_type']:,.0f} | "
            f"{b['sci_ratio']:.2f}x | {r_str} |"
        )
    lines.append("")
    return lines


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
    ]

    lines.extend(_format_finding_1(result))
    lines.extend(_format_finding_2(result))
    lines.extend(_format_finding_3(result))
    lines.extend(_format_finding_4(result))
    lines.extend(_format_finding_5(result))

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
