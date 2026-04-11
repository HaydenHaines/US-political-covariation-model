export const DUSTY_INK = {
  safeD: "#2d4a6f", likelyD: "#4b6d90", leanD: "#7e9ab5",
  tossup: "#b5a995",
  leanR: "#c4907a", likelyR: "#9e5e4e", safeR: "#6e3535",
  background: "#fafaf8", text: "#3a3632", textMuted: "#6e6860",
  textSubtle: "#8a8478", cardBg: "#f5f3ef", border: "#e0ddd8", mapEmpty: "#eae7e2",
} as const;

export type Rating = "safe_d" | "likely_d" | "lean_d" | "tossup" | "lean_r" | "likely_r" | "safe_r";

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

export function ratingColor(rating: Rating): string {
  const map: Record<Rating, string> = {
    safe_d: DUSTY_INK.safeD, likely_d: DUSTY_INK.likelyD, lean_d: DUSTY_INK.leanD,
    tossup: DUSTY_INK.tossup,
    lean_r: DUSTY_INK.leanR, likely_r: DUSTY_INK.likelyR, safe_r: DUSTY_INK.safeR,
  };
  return map[rating];
}

export function ratingLabel(rating: Rating): string {
  const map: Record<Rating, string> = {
    safe_d: "Safe D", likely_d: "Likely D", lean_d: "Lean D",
    tossup: "Tossup",
    lean_r: "Lean R", likely_r: "Likely R", safe_r: "Safe R",
  };
  return map[rating];
}

// Dusty Ink choropleth — delegates to the canonical implementation in palette.ts
export { dustyInkChoropleth } from "@/lib/config/palette";
