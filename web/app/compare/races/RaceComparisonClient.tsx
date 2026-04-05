"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { RaceSelector } from "./RaceSelector";
import { RaceComparisonPanel } from "./RaceComparisonPanel";

// API base: prefer NEXT_PUBLIC_API_URL, fall back to relative path for Next.js rewrites
const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

// ── Types ──────────────────────────────────────────────────────────────────

export interface PollConfidence {
  n_polls: number;
  n_pollsters: number;
  n_methodologies: number;
  label: string;
  tooltip: string;
}

export interface TypeBreakdown {
  type_id: number;
  display_name: string;
  n_counties: number;
  mean_pred_dem_share: number | null;
  total_votes: number | null;
}

export interface HistoricalContext {
  last_race: {
    year: number;
    winner: string;
    party: string;
    margin: number;
    note: string | null;
  };
  presidential_2024: {
    winner: string;
    party: string;
    margin: number;
    note: string | null;
  };
  forecast_shift: number | null;
}

export interface LatestPoll {
  date: string | null;
  pollster: string | null;
  dem_share: number;
  grade: string | null;
}

export interface RaceComparisonData {
  slug: string;
  race: string;
  state_abbr: string;
  race_type: string;
  year: number;
  prediction: number | null;
  pred_std: number | null;
  pred_lo90: number | null;
  pred_hi90: number | null;
  n_counties: number;
  n_polls: number;
  poll_confidence: PollConfidence | null;
  latest_poll: LatestPoll | null;
  type_breakdown: TypeBreakdown[];
  historical_context: HistoricalContext | null;
}

interface ComparisonResponse {
  slugs: [string, string];
  races: [RaceComparisonData, RaceComparisonData];
}

interface RaceOption {
  slug: string;
  label: string;
}

// ── Helpers ────────────────────────────────────────────────────────────────

/**
 * Convert a race slug to a human-readable label.
 * "2026-fl-senate" -> "2026 FL Senate"
 */
function slugToLabel(slug: string): string {
  return slug
    .split("-")
    .map((part, i) => (i === 1 ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(" ");
}

// ── Component ──────────────────────────────────────────────────────────────

/**
 * Race comparison client — reads ?races=slug1,slug2 from the URL,
 * renders two RaceSelector dropdowns, and fetches comparison data
 * whenever both slugs are set.
 *
 * URL state is the single source of truth: changing a selector updates the URL,
 * which triggers a new fetch.  This means the comparison is shareable by link.
 */
export function RaceComparisonClient() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Parse slugs from ?races=slug1,slug2
  const rawRaces = searchParams.get("races") ?? "";
  const slugParts = rawRaces ? rawRaces.split(",").map((s) => s.trim()) : [];
  const slugA = slugParts[0] ?? "";
  const slugB = slugParts[1] ?? "";

  const [raceOptions, setRaceOptions] = useState<RaceOption[]>([]);
  const [comparisonData, setComparisonData] = useState<ComparisonResponse | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the list of available race slugs for the selectors
  useEffect(() => {
    setLoadingOptions(true);
    fetch(`${API_BASE}/forecast/race-slugs`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load races");
        return res.json() as Promise<string[]>;
      })
      .then((slugs) => {
        setRaceOptions(slugs.map((s) => ({ slug: s, label: slugToLabel(s) })));
      })
      .catch(() => {
        // Silently ignore options load failure — selectors will be empty
        setRaceOptions([]);
      })
      .finally(() => setLoadingOptions(false));
  }, []);

  // Fetch comparison data when both slugs are set
  const fetchComparison = useCallback(
    (a: string, b: string) => {
      if (!a || !b) {
        setComparisonData(null);
        setError(null);
        return;
      }
      setLoadingComparison(true);
      setError(null);
      fetch(`${API_BASE}/forecast/compare?slugs=${encodeURIComponent(a)},${encodeURIComponent(b)}`)
        .then((res) => {
          if (res.status === 404) throw new Error("One or both races were not found.");
          if (res.status === 422) throw new Error("Invalid race selection. Please choose two different races.");
          if (!res.ok) throw new Error(`Server error (${res.status})`);
          return res.json() as Promise<ComparisonResponse>;
        })
        .then((data) => setComparisonData(data))
        .catch((err: Error) => setError(err.message))
        .finally(() => setLoadingComparison(false));
    },
    [],
  );

  // Re-fetch when URL slugs change
  useEffect(() => {
    fetchComparison(slugA, slugB);
  }, [slugA, slugB, fetchComparison]);

  // Update URL when a selector changes
  const handleSlugChange = (index: 0 | 1, newSlug: string) => {
    const a = index === 0 ? newSlug : slugA;
    const b = index === 1 ? newSlug : slugB;
    const races = [a, b].filter(Boolean).join(",");
    const params = new URLSearchParams(searchParams.toString());
    if (races) {
      params.set("races", races);
    } else {
      params.delete("races");
    }
    router.push(`/compare/races?${params.toString()}`, { scroll: false });
  };

  const isReady = !!slugA && !!slugB;

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 pb-20">
      {/* Breadcrumb */}
      <nav aria-label="breadcrumb" className="text-xs mb-6" style={{ color: "var(--color-text-muted)" }}>
        <ol className="flex flex-wrap items-center gap-x-1 list-none p-0 m-0">
          <li>
            <Link href="/" style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}>
              Home
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li>
            <Link href="/compare" style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}>
              Compare
            </Link>
          </li>
          <li aria-hidden="true">/</li>
          <li aria-current="page">Race Comparison</li>
        </ol>
      </nav>

      {/* Header */}
      <header className="mb-8">
        <h1
          className="text-3xl font-bold mb-3"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Race Comparison
        </h1>
        <p className="text-sm leading-relaxed max-w-2xl" style={{ color: "var(--color-text-muted)" }}>
          Select two races to compare structural model predictions, polling data, electoral type
          composition, and historical context side by side. The URL updates as you choose races
          — share it to bookmark any comparison.
        </p>
      </header>

      {/* Race selectors */}
      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8 rounded-md p-4"
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
        }}
      >
        <div>
          <label
            htmlFor="race-selector-a"
            className="block text-xs font-semibold mb-1.5 uppercase tracking-wide"
            style={{ color: "var(--color-text-muted)" }}
          >
            Race A
          </label>
          <RaceSelector
            id="race-selector-a"
            options={raceOptions}
            value={slugA}
            placeholder={loadingOptions ? "Loading races…" : "Select a race"}
            disabled={loadingOptions}
            onChange={(slug) => handleSlugChange(0, slug)}
          />
        </div>
        <div>
          <label
            htmlFor="race-selector-b"
            className="block text-xs font-semibold mb-1.5 uppercase tracking-wide"
            style={{ color: "var(--color-text-muted)" }}
          >
            Race B
          </label>
          <RaceSelector
            id="race-selector-b"
            options={raceOptions}
            value={slugB}
            placeholder={loadingOptions ? "Loading races…" : "Select a race"}
            disabled={loadingOptions}
            onChange={(slug) => handleSlugChange(1, slug)}
          />
        </div>
      </div>

      {/* State feedback */}
      {!isReady && !loadingOptions && (
        <div
          className="text-center py-12 rounded-md"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text-muted)",
          }}
        >
          <p className="text-sm">Select two races above to begin the comparison.</p>
        </div>
      )}

      {error && (
        <div
          className="rounded-md px-4 py-3 text-sm mb-6"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--forecast-lean-r)",
            color: "var(--forecast-likely-r)",
          }}
        >
          {error}
        </div>
      )}

      {/* Loading state */}
      {isReady && loadingComparison && (
        <div
          className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          aria-label="Loading comparison"
          aria-busy="true"
        >
          {[0, 1].map((i) => (
            <div
              key={i}
              className="rounded-md p-6 animate-pulse"
              style={{
                background: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                minHeight: "320px",
              }}
            />
          ))}
        </div>
      )}

      {/* Comparison panels */}
      {!loadingComparison && comparisonData && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {comparisonData.races.map((race) => (
            <RaceComparisonPanel key={race.slug} data={race} />
          ))}
        </div>
      )}
    </div>
  );
}
