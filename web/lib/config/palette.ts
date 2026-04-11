/**
 * Color palette configuration for WetherVane.
 *
 * Three color systems:
 * 1. Rating colors — the Dusty Ink v2 partisan scale (7 tiers, purple tossup)
 * 2. Super-type colors — RGB tuples for deck.gl choropleth (8 types + overflow)
 * 3. Choropleth interpolation — continuous dem share -> RGBA mapping
 *
 * Update this file when:
 * - The number of super-types changes after a retrain
 * - The partisan color scale is redesigned
 * - New map visualization modes are added
 */

import type { Rating } from "../types";

// ---------------------------------------------------------------------------
// Dusty Ink v2 — partisan rating scale
// ---------------------------------------------------------------------------

/** Dusty Ink v2 palette: muted, academic, non-garish partisan colors. */
export const DUSTY_INK = {
  safeD:      "#2d4a6f",
  likelyD:    "#4b6d90",
  leanD:      "#7e9ab5",
  tossup:     "#8a6b8a",  // Purple — the signature WetherVane tossup color
  leanR:      "#c4907a",
  likelyR:    "#9e5e4e",
  safeR:      "#6e3535",
  background: "#fafaf8",
  text:       "#3a3632",
  textMuted:  "#6e6860",
  textSubtle: "#8a8478",
  cardBg:     "#f5f3ef",
  border:     "#e0ddd8",
  mapEmpty:   "#eae7e2",
} as const;

/**
 * Dusty Ink partisan palette — authoritative source for all partisan colors in
 * the UI. Components must import from here; no component defines its own
 * partisan colors.
 *
 * Color roles:
 * - DEM_PRIMARY / GOP_PRIMARY  — prediction trend lines and text accents
 * - DEM_SECONDARY / GOP_SECONDARY — scatter dots and softer markers
 * - DEM_SAFE / GOP_SAFE        — safe-seat segments in the balance bar
 * - TOSSUP                      — tossup/neutral marker
 */
export const PALETTE = {
  // Primary partisan colors for trend lines and prediction accents
  DEM_PRIMARY:   "#2166ac",
  GOP_PRIMARY:   "#c0392b",
  // Secondary partisan colors for scatter dots and softer elements
  DEM_SECONDARY: "#4b6d90",   // = DUSTY_INK.likelyD
  GOP_SECONDARY: "#9e5e4e",   // = DUSTY_INK.likelyR
  // Safe-seat shades for the balance bar
  DEM_SAFE:      "#2d4a6f",   // = DUSTY_INK.safeD
  GOP_SAFE:      "#6e3535",   // = DUSTY_INK.safeR
  // Neutral / tossup
  TOSSUP:        "#8a6b8a",   // = DUSTY_INK.tossup
} as const;

/** Map from Rating enum to hex color. */
export const RATING_COLORS: Record<Rating, string> = {
  safe_d:   DUSTY_INK.safeD,
  likely_d: DUSTY_INK.likelyD,
  lean_d:   DUSTY_INK.leanD,
  tossup:   DUSTY_INK.tossup,
  lean_r:   DUSTY_INK.leanR,
  likely_r: DUSTY_INK.likelyR,
  safe_r:   DUSTY_INK.safeR,
};

/** Human-readable labels for ratings. */
export const RATING_LABELS: Record<Rating, string> = {
  safe_d:   "Safe D",
  likely_d: "Likely D",
  lean_d:   "Lean D",
  tossup:   "Tossup",
  lean_r:   "Lean R",
  likely_r: "Likely R",
  safe_r:   "Safe R",
};

// ---------------------------------------------------------------------------
// Super-type colors — RGB tuples for deck.gl
// ---------------------------------------------------------------------------

/**
 * Semantically assigned palette for the 8 tract super-types.
 * Colors are perceptually distinct, non-partisan, and readable on both
 * light and dark backgrounds.
 *
 * Indices correspond to super_type_id values 0-7. Additional overflow
 * slots are defensive — if J changes and new super-types appear, they
 * get distinguishable colors instead of crashing.
 */
export const SUPER_TYPE_COLORS: [number, number, number][] = [
  [220, 120,  55],  // 0: Hispanic Working Community       — warm amber-orange
  [115,  45, 140],  // 1: Black Urban Neighborhood          — deep violet-purple
  [220, 110, 110],  // 2: White Retirement Town             — muted rose-salmon
  [170,  35,  50],  // 3: Rural Evangelical Heartland       — deep crimson
  [ 38, 145, 145],  // 4: Multiracial Outer Suburb          — teal-cyan
  [195, 155,  25],  // 5: Asian-American Professional       — deep gold-amber
  [ 65, 140, 210],  // 6: Affluent White Suburb             — sky blue
  [ 40, 140,  85],  // 7: Urban Knowledge District          — emerald green
  // Overflow slots (model currently uses 0-7 only)
  [140,  86,  75],  // 8
  [227, 119, 194],  // 9
];

/**
 * Get the RGB color for a super-type by ID.
 * Returns neutral gray for invalid IDs, wraps for overflow.
 */
export function getSuperTypeColor(superTypeId: number): [number, number, number] {
  if (superTypeId < 0) return [180, 180, 180];
  return SUPER_TYPE_COLORS[superTypeId % SUPER_TYPE_COLORS.length];
}

/**
 * Convert an RGB tuple to a hex string.
 * Example: [45, 74, 111] -> "#2d4a6f"
 */
export function rgbToHex(rgb: [number, number, number]): string {
  return "#" + rgb.map((c) => c.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------------------
// Choropleth interpolation
// ---------------------------------------------------------------------------

/**
 * Dusty Ink choropleth: continuous dem share (0-1) to RGBA.
 *
 * Uses the 7-tier Dusty Ink partisan scale at full opacity for bold,
 * saturated colors that match the national state-level view.
 *
 * Breakpoints (dem share → color):
 *   ≤0.35  Safe R      #6e3535
 *    0.40  Likely R    #9e5e4e
 *    0.45  Lean R      #c4907a
 *    0.50  Tossup      #8a6b8a
 *    0.55  Lean D      #7e9ab5
 *    0.60  Likely D    #4b6d90
 *   ≥0.65  Safe D      #2d4a6f
 *
 * Values between breakpoints are linearly interpolated.
 */
export function dustyInkChoropleth(demShare: number): [number, number, number, number] {
  // 7 color stops anchored to dem share values — clean red→purple→blue.
  const stops: Array<[number, [number, number, number]]> = [
    [0.35, [110,  53,  53]],  // Safe R
    [0.40, [158,  94,  78]],  // Likely R
    [0.45, [196, 144, 122]],  // Lean R
    [0.50, [138, 107, 138]],  // Tossup — purple
    [0.55, [126, 154, 181]],  // Lean D
    [0.60, [ 75, 109, 144]],  // Likely D
    [0.65, [ 45,  74, 111]],  // Safe D
  ];

  // Clamp to stop range
  const ds = Math.max(stops[0][0], Math.min(stops[stops.length - 1][0], demShare));

  // Find the two surrounding stops and interpolate
  for (let i = 0; i < stops.length - 1; i++) {
    const [lo, cLo] = stops[i];
    const [hi, cHi] = stops[i + 1];
    if (ds <= hi) {
      const t = (ds - lo) / (hi - lo);
      return [
        Math.round(cLo[0] * (1 - t) + cHi[0] * t),
        Math.round(cLo[1] * (1 - t) + cHi[1] * t),
        Math.round(cLo[2] * (1 - t) + cHi[2] * t),
        230,
      ];
    }
  }

  // Fallback (shouldn't reach)
  return [45, 74, 111, 230];
}

/**
 * Whether a dem share falls in the competitive "swing" band (within 8pp of 50/50).
 * These are tracts where small shifts could flip the outcome.
 */
export function isSwingShare(demShare: number): boolean {
  return Math.abs(demShare - 0.5) < 0.08;
}

/** Gold border color for swing tracts: [R, G, B, A]. */
export const SWING_BORDER_COLOR: [number, number, number, number] = [212, 160, 23, 255];

/**
 * Convert a dem share (0-1) to a Rating category.
 * Thresholds: <3pp = tossup, 3-8pp = lean, 8-15pp = likely, 15pp+ = safe.
 */
export function marginToRating(demShare: number): Rating {
  const margin = demShare - 0.5;
  const abs = Math.abs(margin);
  if (abs < 0.03) return "tossup";
  if (margin > 0) {
    if (abs >= 0.15) return "safe_d";
    if (abs >= 0.08) return "likely_d";
    return "lean_d";
  }
  if (abs >= 0.15) return "safe_r";
  if (abs >= 0.08) return "likely_r";
  return "lean_r";
}
