"use client";

/**
 * FitScoreSection — "Moneyball Fit Scores" for a race detail page.
 *
 * Shows which historical candidates have CTOV profiles best matched to the
 * district.  Fit score = dot(career_CTOV, district_W): a higher score means
 * the candidate's historical overperformance pattern aligns with the types of
 * communities that dominate this district.
 *
 * Rendered as a client component because it manages D/R party toggle state
 * and uses SWR for data fetching.
 */

import { useState } from "react";
import Link from "next/link";
import { useFitScores } from "@/lib/hooks/use-fit-scores";
import type { FitScoreEntry } from "@/lib/api";

// ── Design tokens ─────────────────────────────────────────────────────────────

const PARTY_COLORS = {
  D: "var(--color-dem)",
  R: "var(--color-rep)",
} as const;

/** Derive display color — falls back to muted for independent/unknown parties. */
function partyColor(party: string): string {
  return PARTY_COLORS[party as keyof typeof PARTY_COLORS] ?? "var(--color-text-muted)";
}

// ── FitBar ────────────────────────────────────────────────────────────────────

/**
 * Horizontal bar showing fit score relative to the field maximum.
 *
 * Fit scores are small positive dot-product values (e.g. 0.004–0.020).
 * We normalize to [0, 100%] relative to the max in the returned list so
 * the bars communicate relative ranking, not absolute scale.
 */
function FitBar({
  score,
  maxScore,
  party,
}: {
  score: number;
  maxScore: number;
  party: string;
}) {
  const pct = maxScore > 0 ? Math.round((score / maxScore) * 100) : 0;
  const fill = partyColor(party);

  return (
    <div
      style={{
        flex: 1,
        height: "6px",
        borderRadius: "3px",
        background: "var(--color-border)",
        overflow: "hidden",
        minWidth: "40px",
      }}
      aria-hidden="true"
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: fill,
          borderRadius: "3px",
          opacity: 0.75,
          transition: "width 0.25s ease",
        }}
      />
    </div>
  );
}

// ── CandidateRow ──────────────────────────────────────────────────────────────

interface CandidateRowProps {
  entry: FitScoreEntry;
  maxScore: number;
}

function CandidateRow({ entry, maxScore }: CandidateRowProps) {
  const color = partyColor(entry.party);

  return (
    <li
      style={{
        display: "grid",
        // rank | name | bar | score
        gridTemplateColumns: "1.6rem 1fr minmax(60px, 160px) 4.5rem",
        columnGap: "8px",
        alignItems: "center",
        padding: "5px 0",
        borderBottom: "1px solid var(--color-border-subtle)",
      }}
    >
      {/* Rank badge */}
      <span
        aria-label={`Rank ${entry.rank}`}
        style={{
          fontSize: "0.65rem",
          fontWeight: 700,
          color: "var(--color-text-muted)",
          textAlign: "right",
        }}
      >
        #{entry.rank}
      </span>

      {/* Name + metadata */}
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "4px", flexWrap: "wrap" }}>
          <Link
            href={`/candidates/${entry.bioguide_id}`}
            style={{
              fontSize: "0.8rem",
              fontWeight: 600,
              color,
              textDecoration: "none",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
            className="hover:underline"
          >
            {entry.name}
          </Link>
          <span
            style={{
              fontSize: "0.65rem",
              color: "var(--color-text-muted)",
              whiteSpace: "nowrap",
            }}
          >
            ({entry.party}) · {entry.n_races} race{entry.n_races !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      {/* Fit bar */}
      <FitBar score={entry.fit_score} maxScore={maxScore} party={entry.party} />

      {/* Score value */}
      <span
        style={{
          fontSize: "0.7rem",
          fontWeight: 600,
          color: "var(--color-text)",
          textAlign: "right",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {entry.fit_score.toFixed(4)}
      </span>
    </li>
  );
}

// ── PartyToggle ───────────────────────────────────────────────────────────────

interface PartyToggleProps {
  party: "D" | "R";
  onChange: (party: "D" | "R") => void;
}

function PartyToggle({ party, onChange }: PartyToggleProps) {
  const buttonBase: React.CSSProperties = {
    padding: "2px 10px",
    borderRadius: "9999px",
    fontSize: "0.7rem",
    fontWeight: 600,
    cursor: "pointer",
    border: "1px solid",
    transition: "all 0.15s ease",
    letterSpacing: "0.02em",
  };

  const activeD: React.CSSProperties = {
    ...buttonBase,
    background: "rgba(45, 74, 111, 0.15)",
    color: "var(--color-dem)",
    borderColor: "rgba(45, 74, 111, 0.40)",
  };

  const inactiveD: React.CSSProperties = {
    ...buttonBase,
    background: "transparent",
    color: "var(--color-text-muted)",
    borderColor: "var(--color-border)",
  };

  const activeR: React.CSSProperties = {
    ...buttonBase,
    background: "rgba(110, 53, 53, 0.15)",
    color: "var(--color-rep)",
    borderColor: "rgba(110, 53, 53, 0.40)",
  };

  const inactiveR: React.CSSProperties = {
    ...buttonBase,
    background: "transparent",
    color: "var(--color-text-muted)",
    borderColor: "var(--color-border)",
  };

  return (
    <div
      role="group"
      aria-label="Filter fit scores by party"
      style={{ display: "flex", gap: "4px" }}
    >
      <button
        type="button"
        onClick={() => onChange("D")}
        style={party === "D" ? activeD : inactiveD}
        aria-pressed={party === "D"}
      >
        Dem
      </button>
      <button
        type="button"
        onClick={() => onChange("R")}
        style={party === "R" ? activeR : inactiveR}
        aria-pressed={party === "R"}
      >
        Rep
      </button>
    </div>
  );
}

// ── FitScoreSection (public export) ──────────────────────────────────────────

interface FitScoreSectionProps {
  /**
   * Race key in "YYYY ST Office" format, e.g. "2026 GA Senate".
   * Used to query /api/v1/races/{race_key}/fit-scores.
   */
  raceKey: string;
  /**
   * Initial party to display.  Should be set to the party of the leading
   * candidate in the race (e.g. "D" if the Democratic incumbent is running).
   * Falls back to "D" if not provided.
   */
  defaultParty?: "D" | "R";
}

/**
 * Moneyball Fit Scores section for a race detail page.
 *
 * Displays the top 10 historical candidates whose CTOV profiles most closely
 * match the district's electoral type composition.  Users can toggle between
 * Democratic and Republican candidate pools.
 *
 * Layout:
 *   Section header with D/R toggle
 *   Ranked list: # | Name (party · N races) | bar | score
 *   Footer: methodology tooltip
 */
export function FitScoreSection({
  raceKey,
  defaultParty = "D",
}: FitScoreSectionProps) {
  const [party, setParty] = useState<"D" | "R">(defaultParty);
  const { data, isLoading, error } = useFitScores(raceKey, party);

  // Compute the max score in the current list for bar normalization.
  // Slice to top 10 for display.
  const top10 = data?.candidates.slice(0, 10) ?? [];
  const maxScore = top10.reduce((m, c) => Math.max(m, c.fit_score), 0);

  const containerStyle: React.CSSProperties = {
    background: "var(--color-surface)",
    border: "1px solid var(--color-border)",
    borderRadius: "6px",
    padding: "16px",
    marginBottom: "24px",
  };

  const headerStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: "12px",
    flexWrap: "wrap",
    gap: "8px",
  };

  const sectionTitleStyle: React.CSSProperties = {
    fontSize: "0.68rem",
    fontWeight: 600,
    textTransform: "uppercase" as const,
    letterSpacing: "0.06em",
    color: "var(--color-text-muted)",
    margin: 0,
  };

  return (
    <section
      style={containerStyle}
      aria-label="Moneyball Fit Scores"
    >
      {/* Header row: title + D/R toggle */}
      <div style={headerStyle}>
        <h3 style={sectionTitleStyle}>
          Moneyball Fit Scores
        </h3>
        <PartyToggle party={party} onChange={setParty} />
      </div>

      {/* Subtitle / methodology note */}
      <p
        style={{
          fontSize: "0.7rem",
          color: "var(--color-text-muted)",
          marginBottom: "12px",
          lineHeight: 1.4,
        }}
      >
        Historical candidates ranked by how well their CTOV profile matches this district&apos;s
        electoral type composition.
      </p>

      {/* Loading state */}
      {isLoading && (
        <div
          aria-label="Loading fit scores"
          style={{ display: "flex", flexDirection: "column", gap: "8px" }}
        >
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              style={{
                height: "24px",
                borderRadius: "4px",
                background: "var(--color-border)",
                opacity: 0.5 - i * 0.07,
                animation: "pulse 1.5s ease-in-out infinite",
              }}
            />
          ))}
        </div>
      )}

      {/* Error state */}
      {!isLoading && error && (
        <p
          style={{
            fontSize: "0.75rem",
            color: "var(--color-text-muted)",
            textAlign: "center",
            padding: "12px 0",
          }}
        >
          Fit score data unavailable for this race.
        </p>
      )}

      {/* Empty state (no candidates meet min_races threshold) */}
      {!isLoading && !error && data && top10.length === 0 && (
        <p
          style={{
            fontSize: "0.75rem",
            color: "var(--color-text-muted)",
            textAlign: "center",
            padding: "12px 0",
          }}
        >
          No {party === "D" ? "Democratic" : "Republican"} candidates with enough races to rank.
        </p>
      )}

      {/* Ranked list */}
      {!isLoading && !error && top10.length > 0 && (
        <>
          {/* Column headers */}
          <div
            aria-hidden="true"
            style={{
              display: "grid",
              gridTemplateColumns: "1.6rem 1fr minmax(60px, 160px) 4.5rem",
              columnGap: "8px",
              marginBottom: "4px",
            }}
          >
            <span />
            <span
              style={{
                fontSize: "0.6rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "var(--color-text-muted)",
              }}
            >
              Candidate
            </span>
            <span
              style={{
                fontSize: "0.6rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "var(--color-text-muted)",
              }}
            >
              Fit
            </span>
            <span
              style={{
                fontSize: "0.6rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "var(--color-text-muted)",
                textAlign: "right",
              }}
            >
              Score
            </span>
          </div>

          <ul
            style={{ listStyle: "none", margin: 0, padding: 0 }}
            aria-label={`Top ${top10.length} ${party === "D" ? "Democratic" : "Republican"} fit candidates`}
          >
            {top10.map((entry) => (
              <CandidateRow key={entry.bioguide_id} entry={entry} maxScore={maxScore} />
            ))}
          </ul>

          {/* Footer note */}
          <p
            style={{
              fontSize: "0.63rem",
              color: "var(--color-text-muted)",
              marginTop: "10px",
              lineHeight: 1.35,
            }}
          >
            Score = career CTOV · district W. Higher means stronger structural match.
            Requires ≥ 2 races of sabermetric history.
          </p>
        </>
      )}
    </section>
  );
}
