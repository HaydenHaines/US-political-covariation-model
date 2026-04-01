"use client";

import { useRef } from "react";
import Link from "next/link";
import { useScrollZone } from "@/lib/hooks/use-scroll-zone";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { SenateScrollySidebar, SenateScrollySidebarMobile } from "@/components/home/SenateScrollySidebar";
import type { SenateScrollyContextData, SenateRaceData, SenateOverviewData } from "@/lib/api";
import { DUSTY_INK } from "@/lib/config/palette";

interface ScrollyNarrativeProps {
  scrollyData: SenateScrollyContextData;
  overviewData: SenateOverviewData;
}

/**
 * Container for all narrative zones. Renders:
 * - A sticky sidebar (desktop) / horizontal strip (mobile) showing 100 Senate seats
 * - Six narrative zones that track which zone is currently in view
 * - An exit CTA with a full balance bar reprise
 *
 * All numbers come from API data — nothing is hardcoded.
 */
export function ScrollyNarrative({ scrollyData, overviewData }: ScrollyNarrativeProps) {
  const { zone_counts, not_up_d_states, not_up_r_states, structural_context, competitive_races } = scrollyData;
  const sc = structural_context;

  // Refs for each narrative zone section
  const zone1Ref = useRef<HTMLElement>(null);
  const zone2Ref = useRef<HTMLElement>(null);
  const zone3Ref = useRef<HTMLElement>(null);
  const zone4Ref = useRef<HTMLElement>(null);
  const zone5Ref = useRef<HTMLElement>(null);
  const zone6Ref = useRef<HTMLElement>(null);

  const zoneRefs = [zone1Ref, zone2Ref, zone3Ref, zone4Ref, zone5Ref, zone6Ref];
  const activeZone = useScrollZone(zoneRefs);

  // Safe D seats that are on the ballot (not the holdover seats)
  const safeUpD = overviewData.races.filter(
    (r) => r.rating === "safe_d" || r.rating === "likely_d",
  );
  const safeUpDStates = safeUpD.map((r) => r.state);

  // Safe R seats that are on the ballot
  const safeUpR = overviewData.races.filter(
    (r) => r.rating === "safe_r" || r.rating === "likely_r",
  );
  const safeUpRStates = safeUpR.map((r) => r.state);

  // Competitive races: lean_d, tossup, lean_r — sorted by closeness to center
  const battlegroundRaces = competitive_races
    .slice()
    .sort((a, b) => Math.abs(a.margin) - Math.abs(b.margin));

  return (
    <div>
      {/* Mobile horizontal strip — pinned below nav, visible on small screens */}
      <SenateScrollySidebarMobile zoneCounts={zone_counts} activeZone={activeZone} />

      {/* Desktop sidebar + scrollable narrative */}
      <div className="flex gap-6 px-4 max-w-5xl mx-auto">
        {/* Sticky sidebar — desktop only */}
        <SenateScrollySidebar
          zoneCounts={zone_counts}
          activeZone={activeZone}
          className="hidden lg:flex"
        />

        {/* Narrative content */}
        <div className="flex-1 min-w-0">

          {/* ── Zone 1: Not Up (D) ─────────────────────────────────── */}
          <section
            ref={zone1Ref}
            data-zone="not_up_d"
            className="py-20 border-b border-[var(--color-border)]"
          >
            <div className="max-w-2xl">
              <ZoneLabel color={DUSTY_INK.safeD}>Not on the Ballot — Democrats</ZoneLabel>
              <p className="mt-4 text-lg leading-relaxed" style={{ color: "var(--color-text)" }}>
                <strong>{zone_counts.not_up_d ?? 0} Democratic seats</strong> aren&rsquo;t
                on the ballot in 2026. Senate terms run six years, and these senators were
                elected in 2022 or 2024 — they won&rsquo;t face voters again until 2028 at the
                earliest. These seats form the Democratic floor: safe from any wave, immune to
                the political environment, already locked in.
              </p>
              {not_up_d_states.length > 0 && (
                <p className="mt-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
                  {not_up_d_states.join(" · ")}
                </p>
              )}
            </div>
          </section>

          {/* ── Zone 2: Safe D Up ──────────────────────────────────── */}
          <section
            ref={zone2Ref}
            data-zone="safe_up_d"
            className="py-20 border-b border-[var(--color-border)]"
          >
            <div className="max-w-2xl">
              <ZoneLabel color={DUSTY_INK.likelyD}>Safe Democratic Seats Up</ZoneLabel>
              <p className="mt-4 text-lg leading-relaxed" style={{ color: "var(--color-text)" }}>
                These seats were won in the 2020 cycle. The incumbents are expected to hold,
                but they&rsquo;re defending seats won under specific conditions — and the 2026
                environment may look nothing like 2020. Whether that helps or hurts depends on
                how the national mood shifts.
              </p>
              {safeUpDStates.length > 0 && (
                <p className="mt-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
                  <span className="font-medium">Defending:</span>{" "}
                  {safeUpDStates.join(", ")}
                </p>
              )}
            </div>
          </section>

          {/* ── Zone 3: The Structural Problem ─────────────────────── */}
          <section
            ref={zone3Ref}
            data-zone="contested_d"
            className="py-20 border-b border-[var(--color-border)]"
          >
            <div className="max-w-2xl">
              <ZoneLabel color={DUSTY_INK.tossup}>The Structural Landscape</ZoneLabel>
              <p className="mt-4 text-lg leading-relaxed" style={{ color: "var(--color-text)" }}>
                Here&rsquo;s what the map looks like.
              </p>
              {/* Highlighted callout box for the structural argument */}
              <div
                className="mt-6 rounded-lg border-l-4 px-5 py-4"
                style={{
                  borderLeftColor: DUSTY_INK.tossup,
                  background: "var(--color-surface)",
                  borderTop: `1px solid var(--color-border)`,
                  borderRight: `1px solid var(--color-border)`,
                  borderBottom: `1px solid var(--color-border)`,
                }}
              >
                <p className="text-base leading-relaxed" style={{ color: "var(--color-text)" }}>
                  WetherVane&rsquo;s model says that even if Democrats replicate{" "}
                  <strong>{sc.baseline_year}&rsquo;s {sc.baseline_label}</strong> nationally,
                  Republicans still hold the Senate. The model projects Democrats winning{" "}
                  <strong>{sc.dem_wins_at_baseline} of the 33 contested seats</strong> — giving
                  them <strong>{sc.total_dem_projected} total</strong>, still{" "}
                  <strong>{sc.structural_gap} short</strong> of the{" "}
                  {sc.seats_needed_for_majority} needed for a majority.
                </p>
              </div>
              <p className="mt-4 text-base" style={{ color: "var(--color-text-muted)" }}>
                The map favors the party that held more seats going into this cycle. To flip
                control, the challenging party needs to outperform the {sc.baseline_year} baseline
                by a significant margin — not just match it.
              </p>
            </div>
          </section>

          {/* ── Zone 4: Battleground ───────────────────────────────── */}
          <section
            ref={zone4Ref}
            data-zone="tossup"
            className="py-20 border-b border-[var(--color-border)]"
          >
            <div className="max-w-2xl">
              <ZoneLabel color={DUSTY_INK.leanD}>The Battleground</ZoneLabel>
              <p className="mt-4 text-base leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
                Control of the Senate comes down to these races. Every one of them is
                genuinely uncertain. Sorted by margin — the closest calls at the top.
              </p>
            </div>

            {battlegroundRaces.length > 0 ? (
              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl">
                {battlegroundRaces.map((race) => (
                  <BattlegroundRaceCard key={race.slug} race={race} />
                ))}
              </div>
            ) : (
              <p className="mt-4 text-sm italic" style={{ color: "var(--color-text-muted)" }}>
                No competitive races detected.
              </p>
            )}
          </section>

          {/* ── Zone 5: Safe R Up ──────────────────────────────────── */}
          <section
            ref={zone5Ref}
            data-zone="safe_up_r"
            className="py-20 border-b border-[var(--color-border)]"
          >
            <div className="max-w-2xl">
              <ZoneLabel color={DUSTY_INK.likelyR}>Safe Republican Seats Up</ZoneLabel>
              <p className="mt-4 text-lg leading-relaxed" style={{ color: "var(--color-text)" }}>
                These Republican seats held through {sc.baseline_year}&rsquo;s{" "}
                {sc.baseline_label}. The communities that elect these senators are deeply
                conservative — reliable Republican territory across multiple cycles. These seats
                rarely change hands, though upsets do happen in exceptional environments.
              </p>
              {safeUpRStates.length > 0 && (
                <p className="mt-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
                  <span className="font-medium">GOP defending:</span>{" "}
                  {safeUpRStates.join(", ")}
                </p>
              )}
            </div>
          </section>

          {/* ── Zone 6: Not Up (R) ─────────────────────────────────── */}
          <section
            ref={zone6Ref}
            data-zone="not_up_r"
            className="py-20 border-b border-[var(--color-border)]"
          >
            <div className="max-w-2xl">
              <ZoneLabel color={DUSTY_INK.safeR}>Not on the Ballot — Republicans</ZoneLabel>
              <p className="mt-4 text-lg leading-relaxed" style={{ color: "var(--color-text)" }}>
                <strong>{zone_counts.not_up_r ?? 0} Republican seats</strong> are off the
                ballot entirely. No matter what happens in November — wave, surge, or
                collapse — these seats don&rsquo;t move. They are the Republican floor, baked
                in before a single vote is cast in 2026.
              </p>
              {not_up_r_states.length > 0 && (
                <p className="mt-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
                  {not_up_r_states.join(" · ")}
                </p>
              )}
            </div>
          </section>

          {/* ── Exit CTA ───────────────────────────────────────────── */}
          <section className="py-20">
            <div className="max-w-2xl">
              <h2
                className="text-xl font-bold mb-2"
                style={{ color: "var(--color-text)" }}
              >
                Watch the map change
              </h2>
              <p className="text-base mb-6" style={{ color: "var(--color-text-muted)" }}>
                Every new poll shifts the picture. WetherVane updates each race as data
                arrives — propagating signals through the community structure to every state,
                not just the ones being polled. Dig into the individual races, or explore the
                community types the model is actually built on.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <Link
                href="/forecast"
                className="inline-flex items-center rounded-md px-5 py-2.5 text-sm font-semibold transition-opacity hover:opacity-80"
                style={{ background: DUSTY_INK.safeD, color: "white" }}
              >
                All Senate races &rarr;
              </Link>
              <Link
                href="/methodology"
                className="inline-flex items-center rounded-md border px-5 py-2.5 text-sm font-semibold transition-colors hover:bg-[var(--color-surface-raised)]"
                style={{
                  borderColor: "var(--color-border)",
                  color: "var(--color-text)",
                }}
              >
                How it works
              </Link>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

/** Small label above a narrative zone, colored by partisan zone. */
function ZoneLabel({
  color,
  children,
}: {
  color: string;
  children: React.ReactNode;
}) {
  return (
    <p
      className="text-xs font-semibold uppercase tracking-widest mb-1"
      style={{ color }}
    >
      {children}
    </p>
  );
}

/**
 * Compact race card for the battleground zone.
 * Reuses RatingBadge and MarginDisplay from shared components.
 */
function BattlegroundRaceCard({ race }: { race: SenateRaceData }) {
  return (
    <Link
      href={`/forecast/${race.slug}`}
      className="flex items-center justify-between rounded-lg border px-4 py-3 no-underline transition-colors hover:bg-[var(--color-surface-raised)]"
      style={{
        background: "var(--color-surface)",
        borderColor: "var(--color-border)",
      }}
    >
      <div className="flex flex-col gap-1 min-w-0">
        <span
          className="text-sm font-semibold truncate"
          style={{ color: "var(--color-text)" }}
        >
          {race.state}
        </span>
        <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
          {race.n_polls} poll{race.n_polls === 1 ? "" : "s"}
        </span>
      </div>
      <div className="flex items-center gap-2 shrink-0 ml-3">
        <MarginDisplay demShare={race.margin + 0.5} size="sm" />
        <RatingBadge rating={race.rating} />
      </div>
    </Link>
  );
}
