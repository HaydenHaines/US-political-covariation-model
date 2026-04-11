"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Menu, X } from "lucide-react";
import { MAIN_NAV } from "@/lib/config/navigation";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { RATING_COLORS, RATING_LABELS } from "@/lib/config/palette";
import { cn } from "@/lib/utils";
import type { SenateRaceData } from "@/lib/api";
import type { Rating } from "@/lib/types";

/**
 * Number of Dem and GOP holdover seats (not up in 2026).
 * Class I + Class III holdovers: 33D + 34R = 67 total.
 */
const DEM_HOLDOVER_SEATS = 33;
const GOP_HOLDOVER_SEATS = 34;

/**
 * How many Class II seats Dems must win to reach 51 majority seats:
 *   51 target - 33 holdover = 18 wins needed.
 */
const DEM_WINS_NEEDED = 51 - DEM_HOLDOVER_SEATS;

/** Height of each segment in the compact header bar (px). */
const BAR_HEIGHT = 24;

/** Format a margin number as a human-readable string like "D+5.2" or "R+3.1". */
function formatMargin(margin: number): string {
  const pct = Math.abs(margin * 100).toFixed(1);
  if (Math.abs(margin) < 0.001) return "Even";
  return margin >= 0 ? `D+${pct}` : `R+${pct}`;
}

/**
 * Tooltip for a tipping point bar segment.
 *
 * Positioned above the hovered segment using a ref to the segment element.
 * Uses Dusty Ink design system CSS variables for consistent styling.
 */
function SegmentTooltip({
  race,
  segmentRef,
  barRef,
}: {
  race: SenateRaceData;
  segmentRef: HTMLDivElement;
  barRef: HTMLDivElement;
}) {
  const [pos, setPos] = useState<{ left: number; bottom: number } | null>(null);

  useEffect(() => {
    const segRect = segmentRef.getBoundingClientRect();
    const barRect = barRef.getBoundingClientRect();
    // Center tooltip horizontally on the segment, relative to the bar container
    const left = segRect.left - barRect.left + segRect.width / 2;
    // Position above the bar
    const bottom = barRect.height + 6;
    setPos({ left, bottom });
  }, [segmentRef, barRef]);

  if (!pos) return null;

  const label =
    RATING_LABELS[race.rating as Rating] ?? race.rating;
  const color =
    RATING_COLORS[race.rating as Rating] ?? RATING_COLORS.tossup;

  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: pos.left,
        bottom: pos.bottom,
        transform: "translateX(-50%)",
        zIndex: 60,
      }}
    >
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: 6,
          padding: "6px 10px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
          whiteSpace: "nowrap",
          fontSize: 12,
          lineHeight: 1.4,
          color: "var(--color-text)",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 2 }}>{race.race}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: color,
              flexShrink: 0,
            }}
          />
          <span style={{ fontWeight: 600 }}>{label}</span>
          <span style={{ color: "var(--color-text-muted)" }}>
            {formatMargin(race.margin)}
          </span>
        </div>
        {race.n_polls > 0 && (
          <div
            style={{
              fontSize: 10,
              color: "var(--color-text-muted)",
              marginTop: 2,
            }}
          >
            {race.n_polls} poll{race.n_polls !== 1 ? "s" : ""}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Compact tipping point bar for the site header.
 *
 * Shows the 33 contested Senate seats sorted by Dem margin with a tipping
 * point marker. Segments are interactive: hover shows a tooltip with race
 * details, click navigates to the race's forecast page.
 */
function HeaderTippingPointBar({ races }: { races: SenateRaceData[] }) {
  const sorted = [...races].sort((a, b) => b.margin - a.margin);
  const router = useRouter();
  const barRef = useRef<HTMLDivElement>(null);
  const segmentRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [hoveredSlug, setHoveredSlug] = useState<string | null>(null);

  const handleClick = useCallback(
    (slug: string) => {
      router.push(`/forecast/${slug}`);
    },
    [router],
  );

  const hoveredRace = hoveredSlug
    ? sorted.find((r) => r.slug === hoveredSlug) ?? null
    : null;
  const hoveredSegment = hoveredSlug
    ? segmentRefs.current.get(hoveredSlug) ?? null
    : null;

  return (
    <div className="w-full">
      {/* Holdover-seat labels — abbreviated on mobile */}
      <div className="flex justify-between mb-0.5 text-[10px] font-medium leading-tight">
        <span style={{ color: "var(--color-text-muted)" }}>
          <span className="hidden sm:inline">{DEM_HOLDOVER_SEATS}D not up</span>
          <span className="sm:hidden">{DEM_HOLDOVER_SEATS}D</span>
        </span>
        <span style={{ color: "var(--color-text-muted)" }}>
          <span className="hidden sm:inline">{GOP_HOLDOVER_SEATS}R not up</span>
          <span className="sm:hidden">{GOP_HOLDOVER_SEATS}R</span>
        </span>
      </div>

      {/* Wrapper — position:relative anchors the tooltip above the bar */}
      <div ref={barRef} className="relative" onMouseLeave={() => setHoveredSlug(null)}>
        {/* Bar — overflow-hidden clips segment corners to the rounded border */}
        <div
          className="flex w-full rounded-sm overflow-hidden border border-[rgb(var(--color-border))]"
          style={{ height: BAR_HEIGHT }}
        >
          {sorted.map((race, idx) => {
            const isTippingPoint = idx + 1 === DEM_WINS_NEEDED;
            const color =
              RATING_COLORS[race.rating as keyof typeof RATING_COLORS] ??
              RATING_COLORS.tossup;
            const isHovered = hoveredSlug === race.slug;

            return (
              <div
                key={race.slug}
                ref={(el) => {
                  if (el) segmentRefs.current.set(race.slug, el);
                }}
                className="relative flex items-center justify-center min-w-0"
                style={{
                  flex: 1,
                  height: BAR_HEIGHT,
                  backgroundColor: color,
                  cursor: "pointer",
                  opacity: hoveredSlug && !isHovered ? 0.65 : 1,
                  transition: "opacity 0.12s ease",
                  ...(isTippingPoint
                    ? { borderRight: "3px solid #111111", zIndex: 1 }
                    : {}),
                }}
                onMouseEnter={() => setHoveredSlug(race.slug)}
                onClick={() => handleClick(race.slug)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    handleClick(race.slug);
                  }
                }}
                aria-label={`${race.race}: ${RATING_LABELS[race.rating as Rating] ?? race.rating}, ${formatMargin(race.margin)}. Click to view forecast.`}
              >
                {/* State abbreviation — hidden on narrow viewports */}
                <span
                  className="pointer-events-none select-none hidden md:inline"
                  style={{
                    fontSize: "8px",
                    fontWeight: 600,
                    color: "rgba(255,255,255,0.85)",
                    lineHeight: 1,
                    letterSpacing: "0.02em",
                  }}
                  aria-hidden="true"
                >
                  {race.state}
                </span>

                {/* Tipping point marker */}
                {isTippingPoint && (
                  <span
                    className="absolute pointer-events-none select-none"
                    style={{
                      bottom: BAR_HEIGHT + 1,
                      right: -18,
                      fontSize: "8px",
                      fontWeight: 700,
                      color: "#111111",
                      whiteSpace: "nowrap",
                      lineHeight: 1,
                      zIndex: 2,
                    }}
                    aria-hidden="true"
                  >
                    Maj.
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Tooltip — rendered outside the bar's overflow-hidden container */}
        {hoveredRace && hoveredSegment && barRef.current && (
          <SegmentTooltip
            race={hoveredRace}
            segmentRef={hoveredSegment}
            barRef={barRef.current}
          />
        )}
      </div>
    </div>
  );
}

/**
 * Site header combining navigation and the tipping point bar.
 *
 * Desktop (>=768px): single sticky header with logo + nav links + theme toggle
 * on top row, tipping point bar below.
 * Mobile (<768px): logo + theme toggle + hamburger on top row, compact tipping
 * point bar below. Hamburger opens the slide-in nav panel.
 */
export function SiteHeader() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { data } = useSenateOverview();

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[var(--color-border)] bg-[var(--color-bg)]/80 backdrop-blur-sm">
        {/* Top row: logo + nav */}
        <nav className="mx-auto flex h-10 max-w-5xl items-center justify-between px-4">
          <Link
            href="/"
            className="font-serif text-lg font-black tracking-tight no-underline"
            style={{ color: "var(--color-text)" }}
          >
            WetherVane
          </Link>

          {/* Desktop nav links (>=768px) */}
          <div className="hidden md:flex items-center gap-1">
            {MAIN_NAV.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "rounded px-3 py-1 text-sm no-underline transition-colors",
                    isActive
                      ? "font-semibold"
                      : "hover:bg-[var(--color-surface-raised)]",
                  )}
                  style={{
                    color: isActive
                      ? "var(--color-text)"
                      : "var(--color-text-muted)",
                  }}
                >
                  {item.label}
                </Link>
              );
            })}

            <div className="ml-2 border-l border-[var(--color-border)] pl-2">
              <ThemeToggle />
            </div>
          </div>

          {/* Mobile: theme toggle + hamburger (<768px) */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            <button
              onClick={() => setMobileOpen(true)}
              className="flex items-center justify-center rounded-md p-2 min-h-[44px] min-w-[44px]"
              style={{ color: "var(--color-text)" }}
              aria-label="Open navigation menu"
              aria-expanded={mobileOpen}
            >
              <Menu size={20} aria-hidden />
            </button>
          </div>
        </nav>

        {/* Tipping point bar — compact, below the nav row */}
        {data?.races && data.races.length > 0 && (
          <div className="mx-auto max-w-5xl px-4 pb-1.5">
            <HeaderTippingPointBar races={data.races} />
          </div>
        )}
      </header>

      {/* Mobile: slide-in nav panel (<768px) */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-50 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-modal="true"
          role="dialog"
          aria-label="Navigation menu"
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/40" />

          {/* Panel — slides in from the right */}
          <nav
            className="absolute top-0 right-0 bottom-0 w-64 flex flex-col py-4"
            style={{
              background: "var(--color-bg)",
              borderLeft: "1px solid var(--color-border)",
              boxShadow: "-4px 0 24px rgba(0,0,0,0.15)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 mb-4">
              <span
                className="font-serif text-lg font-black"
                style={{ color: "var(--color-text)" }}
              >
                WetherVane
              </span>
              <button
                onClick={() => setMobileOpen(false)}
                className="p-2 min-h-[44px] min-w-[44px] flex items-center justify-center rounded-md"
                style={{ color: "var(--color-text-muted)" }}
                aria-label="Close navigation menu"
              >
                <X size={20} aria-hidden />
              </button>
            </div>

            {/* Nav links */}
            {MAIN_NAV.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "block px-4 py-3 text-base no-underline transition-colors min-h-[44px]",
                    isActive
                      ? "font-semibold"
                      : "hover:bg-[var(--color-surface-raised)]",
                  )}
                  style={{
                    color: isActive
                      ? "var(--color-text)"
                      : "var(--color-text-muted)",
                  }}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      )}
    </>
  );
}
