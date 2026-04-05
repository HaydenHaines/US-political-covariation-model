"use client";

/**
 * CountySearch — floating search overlay for the stained-glass map.
 *
 * Lets users navigate to any county by name without manually panning/zooming.
 * On selection, the map flies to the county's bounding box using the same
 * FlyToInterpolator pattern used for state navigation.
 *
 * Design: Dusty Ink system — muted surface, serif label, --font-sans input.
 * Position: top-right, below the HistoricalOverlayToggle row (offset to avoid
 * overlap with the History pill buttons that sit at top: 12, right: 54).
 *
 * County data is passed as a prop (extracted from the already-loaded
 * countyGeo GeoJSON that MapShell holds in state) so this component never
 * triggers its own network requests.
 */

import { useState, useRef, useCallback, useId, useEffect } from "react";
import { DUSTY_INK } from "@/lib/config/palette";

// ---------------------------------------------------------------------------
// Constants (Dusty Ink design system values)
// ---------------------------------------------------------------------------

/** Maximum number of dropdown suggestions shown at once. */
const MAX_SUGGESTIONS = 8;

/** Vertical offset from top to avoid the History toggle row. */
const TOP_OFFSET = 48;

/** Right offset matching the History toggle alignment. */
const RIGHT_OFFSET = 54;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Minimal county record extracted from GeoJSON features. */
export interface CountyEntry {
  /** 5-digit FIPS code as a string (e.g. "13121"). */
  fips: string;
  /**
   * Display label combining county name and state abbreviation.
   * Sourced directly from county GeoJSON's county_name property,
   * which already uses "County Name, ST" format (e.g. "Fulton County, GA").
   */
  label: string;
  /** Raw GeoJSON geometry object for bbox computation. */
  geometry: { coordinates: unknown };
}

interface CountySearchProps {
  /**
   * County entries derived from the countyGeo GeoJSON that MapShell has in
   * memory. Pass null/undefined when the GeoJSON has not yet loaded; the
   * component renders nothing in that case to avoid triggering an independent
   * fetch.
   */
  counties: CountyEntry[] | null;
  /** Called when the user selects a county so the map can fly to it. */
  onSelectCounty: (county: CountyEntry) => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Return true if `query` is a substring match against `label` (case-insensitive).
 * Simple substring is sufficient — a full fuzzy matcher would be over-engineered
 * for county names that users can partially type accurately.
 */
function matchesQuery(label: string, query: string): boolean {
  return label.toLowerCase().includes(query.toLowerCase());
}

/**
 * Filter and rank counties by query string.
 * Prefix matches sort before substring matches so "Ful" → "Fulton" rises
 * to the top even when "Blissful County" also matches.
 */
function filterCounties(counties: CountyEntry[], query: string): CountyEntry[] {
  if (!query.trim()) return [];

  const q = query.toLowerCase();
  const matches = counties.filter((c) => matchesQuery(c.label, q));

  // Sort: prefix match first, then alphabetical within each tier
  matches.sort((a, b) => {
    const aPrefix = a.label.toLowerCase().startsWith(q);
    const bPrefix = b.label.toLowerCase().startsWith(q);
    if (aPrefix && !bPrefix) return -1;
    if (!aPrefix && bPrefix) return 1;
    return a.label.localeCompare(b.label);
  });

  return matches.slice(0, MAX_SUGGESTIONS);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const CONTAINER_STYLE: React.CSSProperties = {
  position: "absolute",
  top: TOP_OFFSET,
  right: RIGHT_OFFSET,
  zIndex: 10,
  width: 220,
};

const INPUT_STYLE: React.CSSProperties = {
  width: "100%",
  boxSizing: "border-box",
  padding: "5px 10px",
  borderRadius: 6,
  border: `1px solid var(--color-border, ${DUSTY_INK.border})`,
  background: `var(--color-surface, ${DUSTY_INK.background})`,
  color: `var(--color-text, ${DUSTY_INK.text})`,
  fontSize: 12,
  fontFamily: "var(--font-sans)",
  outline: "none",
  boxShadow: "0 1px 3px rgba(58,54,50,0.08)",
};

const DROPDOWN_STYLE: React.CSSProperties = {
  position: "absolute",
  top: "calc(100% + 2px)",
  left: 0,
  right: 0,
  background: `var(--color-surface, ${DUSTY_INK.background})`,
  border: `1px solid var(--color-border, ${DUSTY_INK.border})`,
  borderRadius: 6,
  boxShadow: "0 4px 12px rgba(58,54,50,0.12)",
  overflow: "hidden",
  // Stacking: must sit above the map canvas (z-index 0) and map overlays (z-index 10)
  zIndex: 11,
};

function SuggestionItem({
  county,
  isHighlighted,
  onSelect,
  onMouseEnter,
  id,
}: {
  county: CountyEntry;
  isHighlighted: boolean;
  onSelect: (c: CountyEntry) => void;
  onMouseEnter: () => void;
  id: string;
}) {
  return (
    <div
      id={id}
      role="option"
      aria-selected={isHighlighted}
      onMouseDown={(e) => {
        // Prevent the input from losing focus before the click fires
        e.preventDefault();
        onSelect(county);
      }}
      onMouseEnter={onMouseEnter}
      style={{
        padding: "6px 10px",
        fontSize: 12,
        fontFamily: "var(--font-sans)",
        color: isHighlighted
          ? `var(--color-text, ${DUSTY_INK.text})`
          : `var(--color-text-muted, ${DUSTY_INK.textMuted})`,
        background: isHighlighted
          ? `var(--color-border, ${DUSTY_INK.border})`
          : "transparent",
        cursor: "pointer",
        lineHeight: "18px",
      }}
    >
      {county.label}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function CountySearch({ counties, onSelectCounty }: CountySearchProps) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<CountyEntry[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  // Keyboard-navigation cursor — -1 means no item is highlighted
  const [cursorIndex, setCursorIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listboxId = useId();

  // Recompute suggestions whenever the query or county list changes.
  // Done in a useEffect so the filter runs asynchronously, keeping keystroke
  // latency below 16ms even on slower devices with the full 3,235-county set.
  useEffect(() => {
    if (!counties || !query.trim()) {
      setSuggestions([]);
      setIsOpen(false);
      return;
    }
    const filtered = filterCounties(counties, query);
    setSuggestions(filtered);
    setIsOpen(filtered.length > 0);
    setCursorIndex(-1);
  }, [query, counties]);

  const handleSelect = useCallback(
    (county: CountyEntry) => {
      setQuery("");
      setSuggestions([]);
      setIsOpen(false);
      setCursorIndex(-1);
      onSelectCounty(county);
    },
    [onSelectCounty]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (!isOpen) return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setCursorIndex((i) => Math.min(i + 1, suggestions.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setCursorIndex((i) => Math.max(i - 1, -1));
      } else if (e.key === "Enter" && cursorIndex >= 0) {
        e.preventDefault();
        handleSelect(suggestions[cursorIndex]);
      } else if (e.key === "Escape") {
        setIsOpen(false);
        setCursorIndex(-1);
      }
    },
    [isOpen, suggestions, cursorIndex, handleSelect]
  );

  // Don't render until county data is available (avoids a blank search box
  // that silently returns no results when the user starts typing early).
  if (!counties) return null;

  const activeDescendantId =
    cursorIndex >= 0 ? `${listboxId}-option-${cursorIndex}` : undefined;

  return (
    <div style={CONTAINER_STYLE} aria-label="County search">
      <input
        ref={inputRef}
        type="search"
        placeholder="Search county…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => {
          if (suggestions.length > 0) setIsOpen(true);
        }}
        onBlur={() => {
          // Delay close slightly so onMouseDown on a suggestion fires first
          setTimeout(() => setIsOpen(false), 150);
        }}
        aria-label="Search for a county"
        aria-autocomplete="list"
        aria-controls={listboxId}
        aria-activedescendant={activeDescendantId}
        aria-expanded={isOpen}
        role="combobox"
        style={INPUT_STYLE}
        autoComplete="off"
        spellCheck={false}
      />

      {isOpen && (
        <div
          id={listboxId}
          role="listbox"
          aria-label="County suggestions"
          style={DROPDOWN_STYLE}
        >
          {suggestions.map((county, i) => (
            <SuggestionItem
              key={county.fips}
              id={`${listboxId}-option-${i}`}
              county={county}
              isHighlighted={i === cursorIndex}
              onSelect={handleSelect}
              onMouseEnter={() => setCursorIndex(i)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
