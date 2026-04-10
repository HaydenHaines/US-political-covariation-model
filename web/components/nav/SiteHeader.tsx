"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { MAIN_NAV } from "@/lib/config/navigation";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { RATING_COLORS } from "@/lib/config/palette";
import { cn } from "@/lib/utils";
import type { SenateRaceData } from "@/lib/api";

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

/**
 * Compact tipping point bar for the site header.
 *
 * Shows the 33 contested Senate seats sorted by Dem margin with a tipping
 * point marker, but without heading text or description paragraph — just
 * the colored segments with state labels and holdover seat counts.
 */
function HeaderTippingPointBar({ races }: { races: SenateRaceData[] }) {
  const sorted = [...races].sort((a, b) => b.margin - a.margin);

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

      {/* Bar */}
      <div
        className="flex w-full rounded-sm overflow-hidden border border-[rgb(var(--color-border))]"
        style={{ height: BAR_HEIGHT }}
      >
        {sorted.map((race, idx) => {
          const isTippingPoint = idx + 1 === DEM_WINS_NEEDED;
          const color =
            RATING_COLORS[race.rating as keyof typeof RATING_COLORS] ??
            RATING_COLORS.tossup;

          return (
            <div
              key={race.slug}
              className="relative flex items-center justify-center min-w-0"
              style={{
                flex: 1,
                height: BAR_HEIGHT,
                backgroundColor: color,
                ...(isTippingPoint
                  ? { borderRight: "3px solid #111111", zIndex: 1 }
                  : {}),
              }}
              title={`${race.state}: ${race.margin >= 0 ? "D+" : "R+"}${(Math.abs(race.margin) * 100).toFixed(1)} (${race.rating})`}
              aria-label={`${race.state}: ${race.rating}`}
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
