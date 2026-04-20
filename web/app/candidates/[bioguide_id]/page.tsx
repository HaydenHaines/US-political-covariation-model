"use client";

/**
 * Candidate profile page (/candidates/[bioguide_id]).
 *
 * Shows:
 * - W-L record (wins and losses from races[])
 * - Election history table (year, state, office, result, dem% vs party avg)
 * - Badge pills (reused from CandidateBadges component's BadgePill logic)
 * - CTOV radar chart (reused CTOVRadarChart component)
 * - Predecessor comparison for single-race candidates (low-trust, clearly labeled)
 * - Fundraising placeholder (Phase 5)
 */

import Link from "next/link";
import {
  useCandidateProfile,
  useCandidatePredecessor,
} from "@/lib/hooks/use-candidate-profile";
import { CTOVRadarChart } from "@/components/forecast/CTOVRadarChart";
import { CandidateStatsCard } from "@/components/forecast/CandidateStatsCard";
import type { RaceResult, CandidateBadge } from "@/lib/api";

// ── Helpers ──────────────────────────────────────────────────────────────────

const PARTY_COLORS: Record<string, string> = {
  D: "var(--color-dem)",
  R: "var(--color-rep)",
};

function partyColor(party: string): string {
  return PARTY_COLORS[party] ?? "var(--color-text-muted)";
}

function formatPct(share: number | null): string {
  if (share === null) return "—";
  return `${(share * 100).toFixed(1)}%`;
}

function formatCEC(cec: number): string {
  return `${Math.round(cec * 100)}%`;
}

// ── Badge pill (simplified version from CandidateBadges.tsx) ─────────────────

const BADGE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  "Hispanic Appeal": {
    bg: "rgba(234, 88, 12, 0.10)",
    text: "rgb(180, 70, 10)",
    border: "rgba(234, 88, 12, 0.30)",
  },
  "Black Community Strength": {
    bg: "rgba(124, 58, 237, 0.10)",
    text: "rgb(100, 45, 190)",
    border: "rgba(124, 58, 237, 0.30)",
  },
  "Senior Whisperer": {
    bg: "rgba(20, 120, 140, 0.10)",
    text: "rgb(15, 95, 115)",
    border: "rgba(20, 120, 140, 0.30)",
  },
  "Suburban Professional": {
    bg: "rgba(59, 130, 246, 0.10)",
    text: "rgb(40, 95, 190)",
    border: "rgba(59, 130, 246, 0.30)",
  },
  "Rural Populist": {
    bg: "rgba(161, 98, 7, 0.10)",
    text: "rgb(130, 80, 5)",
    border: "rgba(161, 98, 7, 0.30)",
  },
  "Faith Coalition": {
    bg: "rgba(168, 85, 120, 0.10)",
    text: "rgb(140, 65, 100)",
    border: "rgba(168, 85, 120, 0.30)",
  },
  "Turnout Monster": {
    bg: "rgba(34, 197, 94, 0.10)",
    text: "rgb(20, 130, 65)",
    border: "rgba(34, 197, 94, 0.30)",
  },
};

const DEFAULT_BADGE_COLOR = {
  bg: "rgba(148, 163, 184, 0.10)",
  text: "rgb(85, 95, 110)",
  border: "rgba(148, 163, 184, 0.30)",
};

function getBadgeColor(badgeName: string) {
  if (badgeName in BADGE_COLORS) return BADGE_COLORS[badgeName];
  if (badgeName.startsWith("Low ")) {
    const base = badgeName.slice(4);
    const baseColor = BADGE_COLORS[base];
    if (baseColor) {
      return {
        bg: baseColor.bg.replace("0.10", "0.06"),
        text: baseColor.text,
        border: baseColor.border.replace("0.30", "0.18"),
      };
    }
  }
  return DEFAULT_BADGE_COLOR;
}

function BadgePill({ badge }: { badge: CandidateBadge }) {
  const isSignature = badge.kind === "signature";
  const isLow = badge.name.startsWith("Low ");
  const color = isSignature ? DEFAULT_BADGE_COLOR : getBadgeColor(badge.name);

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "3px",
        padding: "2px 10px",
        borderRadius: "9999px",
        fontSize: "0.7rem",
        fontWeight: 500,
        letterSpacing: "0.02em",
        background: color.bg,
        color: color.text,
        border: `${badge.provisional ? "1px dashed" : "1px solid"} ${color.border}`,
        opacity: isLow ? 0.8 : 1,
      }}
    >
      {isLow && !isSignature && (
        <span aria-hidden="true" style={{ fontSize: "0.6rem" }}>▽</span>
      )}
      {badge.name}
      {badge.provisional && (
        <span style={{ fontSize: "0.6rem", opacity: 0.7 }}> *</span>
      )}
    </span>
  );
}

// ── Election history table ────────────────────────────────────────────────────

function ElectionHistoryTable({ races }: { races: RaceResult[] }) {
  if (races.length === 0) {
    return (
      <p style={{ fontSize: "0.82rem", color: "var(--color-text-muted)" }}>
        No election history available.
      </p>
    );
  }

  // Compute party average dem share across all races for comparison column.
  const withShares = races.filter((r) => r.actual_dem_share_2party !== null);
  const partyAvg =
    withShares.length > 0
      ? withShares.reduce((sum, r) => sum + (r.actual_dem_share_2party ?? 0), 0) /
        withShares.length
      : null;

  return (
    <table
      style={{
        width: "100%",
        fontSize: "0.78rem",
        borderCollapse: "collapse",
        color: "var(--color-text)",
      }}
    >
      <thead>
        <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
          {["Year", "State", "Office", "Result", "Dem%", "vs Avg"].map((h) => (
            <th
              key={h}
              style={{
                textAlign: "left",
                padding: "4px 8px",
                fontWeight: 600,
                fontSize: "0.68rem",
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {races.map((race, idx) => {
          const share = race.actual_dem_share_2party;
          const delta =
            partyAvg !== null && share !== null ? share - partyAvg : null;
          const resultColor =
            race.result === "win" ? "rgb(20, 130, 65)" : "rgb(200, 60, 40)";

          return (
            <tr
              key={idx}
              style={{
                borderBottom: "1px solid var(--color-border-subtle)",
              }}
            >
              <td style={{ padding: "5px 8px" }}>{race.year}</td>
              <td style={{ padding: "5px 8px" }}>{race.state}</td>
              <td style={{ padding: "5px 8px" }}>{race.office}</td>
              <td
                style={{
                  padding: "5px 8px",
                  fontWeight: 600,
                  color: resultColor,
                  textTransform: "capitalize",
                }}
              >
                {race.result}
              </td>
              <td style={{ padding: "5px 8px" }}>{formatPct(share)}</td>
              <td
                style={{
                  padding: "5px 8px",
                  color:
                    delta === null
                      ? "var(--color-text-muted)"
                      : delta > 0
                      ? "rgb(20, 130, 65)"
                      : "rgb(200, 60, 40)",
                  fontWeight: delta !== null ? 600 : 400,
                }}
              >
                {delta === null
                  ? "—"
                  : `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)}pp`}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ── W-L record display ────────────────────────────────────────────────────────

function WLRecord({ races }: { races: RaceResult[] }) {
  const wins = races.filter((r) => r.result === "win").length;
  const losses = races.filter((r) => r.result === "loss").length;

  return (
    <div
      style={{
        display: "flex",
        gap: "16px",
        alignItems: "center",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            fontSize: "1.8rem",
            fontWeight: 700,
            fontFamily: "var(--font-serif)",
            color: "rgb(20, 130, 65)",
            lineHeight: 1,
          }}
        >
          {wins}
        </div>
        <div style={{ fontSize: "0.65rem", color: "var(--color-text-muted)", marginTop: "2px" }}>
          WIN
        </div>
      </div>
      <div style={{ fontSize: "1.2rem", color: "var(--color-text-muted)" }}>–</div>
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            fontSize: "1.8rem",
            fontWeight: 700,
            fontFamily: "var(--font-serif)",
            color: "rgb(200, 60, 40)",
            lineHeight: 1,
          }}
        >
          {losses}
        </div>
        <div style={{ fontSize: "0.65rem", color: "var(--color-text-muted)", marginTop: "2px" }}>
          LOSS
        </div>
      </div>
    </div>
  );
}

// ── Section wrapper ───────────────────────────────────────────────────────────

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "6px",
        padding: "16px",
        marginBottom: "12px",
      }}
    >
      <h2
        style={{
          fontSize: "0.68rem",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          color: "var(--color-text-muted)",
          marginBottom: "12px",
        }}
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

// ── Predecessor comparison section ────────────────────────────────────────────

function PredecessorSection({
  bioguideId,
  nRaces,
}: {
  bioguideId: string;
  nRaces: number;
}) {
  const { data: predecessor, isLoading } = useCandidatePredecessor(
    bioguideId,
    nRaces,
  );

  // Only show for single-race candidates
  if (nRaces !== 1) return null;
  if (isLoading) return null;
  if (!predecessor) return null;

  return (
    <Section title="Predecessor Comparison (low-trust signal)">
      <p
        style={{
          fontSize: "0.78rem",
          color: "var(--color-text-muted)",
          marginBottom: "8px",
          fontStyle: "italic",
        }}
      >
        This candidate has only one race on record. The closest predecessor in the same
        state/office/party slot provides a weak second data point — not a reliable
        consistency measure.
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span
          style={{
            fontSize: "0.78rem",
            color: "var(--color-text-muted)",
          }}
        >
          vs predecessor:
        </span>
        <Link
          href={`/candidates/${predecessor.bioguide_id}`}
          style={{
            fontSize: "0.85rem",
            fontWeight: 600,
            color: "var(--color-focus)",
            textDecoration: "none",
          }}
          className="hover:underline"
        >
          {predecessor.name}
        </Link>
        <span style={{ fontSize: "0.72rem", color: "var(--color-text-subtle)" }}>
          ({predecessor.year})
        </span>
      </div>
    </Section>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

interface PageProps {
  params: { bioguide_id: string };
}

export default function CandidateProfilePage({ params }: PageProps) {
  const { bioguide_id } = params;
  const { data, isLoading, error } = useCandidateProfile(bioguide_id);

  if (error) {
    return (
      <div style={{ maxWidth: "800px", margin: "0 auto", padding: "24px 16px" }}>
        <Link
          href="/candidates"
          style={{ fontSize: "0.78rem", color: "var(--color-text-muted)" }}
        >
          ← Back to Candidates
        </Link>
        <div
          style={{
            marginTop: "24px",
            padding: "16px",
            borderRadius: "6px",
            background: "rgba(239, 68, 68, 0.08)",
            border: "1px solid rgba(239, 68, 68, 0.25)",
            color: "var(--color-text)",
            fontSize: "0.85rem",
          }}
        >
          Candidate not found or failed to load.
        </div>
      </div>
    );
  }

  if (isLoading || !data) {
    return (
      <div style={{ maxWidth: "800px", margin: "0 auto", padding: "24px 16px" }}>
        <div
          style={{
            height: "24px",
            width: "120px",
            borderRadius: "4px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            marginBottom: "16px",
            animation: "pulse 1.5s ease-in-out infinite",
          }}
        />
        {[200, 180, 280].map((w, i) => (
          <div
            key={i}
            style={{
              height: "80px",
              borderRadius: "6px",
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              marginBottom: "10px",
              animation: "pulse 1.5s ease-in-out infinite",
            }}
          />
        ))}
      </div>
    );
  }

  const color = partyColor(data.party);

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "24px 16px" }}>
      {/* Breadcrumb */}
      <Link
        href="/candidates"
        style={{
          fontSize: "0.78rem",
          color: "var(--color-text-muted)",
          textDecoration: "none",
        }}
        className="hover:underline"
      >
        ← All Candidates
      </Link>

      {/* Header */}
      <div
        style={{
          marginTop: "16px",
          marginBottom: "20px",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div>
          <h1
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: "1.7rem",
              fontWeight: 700,
              color,
              marginBottom: "4px",
            }}
          >
            {data.name}
          </h1>
          <div
            style={{
              fontSize: "0.8rem",
              color: "var(--color-text-muted)",
              display: "flex",
              gap: "8px",
              flexWrap: "wrap",
            }}
          >
            <span>{data.party === "D" ? "Democrat" : data.party === "R" ? "Republican" : data.party}</span>
            <span>·</span>
            <span>{data.n_races} race{data.n_races !== 1 ? "s" : ""} in sabermetric record</span>
            <span>·</span>
            <span>CEC {formatCEC(data.cec)}</span>
          </div>
        </div>

        {/* W-L record */}
        <WLRecord races={data.races} />
      </div>

      {/* Election history */}
      <Section title="Election History">
        <ElectionHistoryTable races={data.races} />
      </Section>

      {/* Predecessor comparison (single-race only) */}
      <PredecessorSection bioguideId={bioguide_id} nRaces={data.n_races} />

      {/* Badges + CTOV radar */}
      {data.badges.length > 0 && (
        <Section title="Performance Badges">
          <div style={{ display: "flex", gap: "16px", alignItems: "flex-start", flexWrap: "wrap" }}>
            {/* Radar chart — padding compensates for label overflow outside SVG bounds */}
            {data.badge_scores && Object.keys(data.badge_scores).length > 0 && (
              <div style={{ flexShrink: 0, padding: "20px", overflow: "visible" }}>
                <CTOVRadarChart
                  badgeScores={data.badge_scores}
                  party={data.party}
                  size={160}
                />
              </div>
            )}

            {/* Badge pills */}
            <div>
              <div
                style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "10px" }}
              >
                {data.badges.map((badge) => (
                  <BadgePill key={badge.name} badge={badge} />
                ))}
              </div>
              <p
                style={{
                  fontSize: "0.7rem",
                  color: "var(--color-text-subtle)",
                  fontStyle: "italic",
                  marginTop: "4px",
                }}
              >
                Badges show performance patterns relative to same-party candidates.
                {data.n_races === 1 && " * = provisional (single-race)"}
              </p>
            </div>
          </div>
        </Section>
      )}

      {/* Campaign Finance + Legislative Profile (Phase 5) */}
      <CandidateStatsCard
        campaignStats={data.campaign_stats}
        legislativeStats={data.legislative_stats}
        party={data.party}
      />
    </div>
  );
}
