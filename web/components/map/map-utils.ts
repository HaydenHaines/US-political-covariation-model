/**
 * Utility functions for the WetherVane map components.
 *
 * Pure functions only — no React, no side effects.
 * Safe to import from both client and server contexts.
 */

/** Format income as "$XX,XXX". Returns empty string for null/undefined. */
export function formatIncome(income: number | null | undefined): string {
  if (income == null) return "";
  return `$${Math.round(income).toLocaleString("en-US")}`;
}

/** Format a decimal fraction (0–1) as "XX%". Returns empty string for null/undefined. */
export function formatPct(val: number | null | undefined): string {
  if (val == null) return "";
  return `${Math.round(val * 100)}%`;
}

/** Parse a hex color string (#rrggbb) to [r, g, b]. */
export function hexToRgb(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

/**
 * Compute bounding box [minLng, minLat, maxLng, maxLat] from a GeoJSON geometry.
 * Recursively walks nested coordinate arrays.
 */
export function bboxFromGeometry(
  geometry: { coordinates: unknown }
): [number, number, number, number] {
  let minLng = Infinity, maxLng = -Infinity;
  let minLat = Infinity, maxLat = -Infinity;

  const recurse = (arr: unknown): void => {
    if (!Array.isArray(arr)) return;
    if (typeof arr[0] === "number") {
      const [lng, lat] = arr as [number, number];
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      return;
    }
    arr.forEach(recurse);
  };

  recurse(geometry.coordinates);
  return [minLng, minLat, maxLng, maxLat];
}

/**
 * Compute a reasonable zoom level from a bounding box span.
 * Clamps between 4 (continental US) and 9 (city level).
 */
export function zoomFromBbox(
  minLng: number,
  minLat: number,
  maxLng: number,
  maxLat: number
): number {
  const lngSpan = maxLng - minLng;
  const latSpan = maxLat - minLat;
  const span = Math.max(lngSpan, latSpan);
  // log2(360/span) gives roughly the zoom for that span to fill the viewport
  return Math.min(9, Math.max(4, Math.log2(360 / span) - 0.5));
}

/** Initial view state: centered on continental US at zoom 3.5 to avoid cropping NE/West edges. */
export const INITIAL_VIEW_STATE = {
  longitude: -98.0,
  latitude: 38.5,
  zoom: 3.5,
  pitch: 0,
  bearing: 0,
} as const;

/** Fly-to transition duration in milliseconds. */
export const TRANSITION_MS = 800;
