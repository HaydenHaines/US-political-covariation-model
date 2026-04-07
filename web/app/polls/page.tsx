import type { Metadata } from "next";
import Link from "next/link";
import { PollsTable, type PollEntry } from "./PollsTable";

// ── Metadata ───────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "2026 Election Polls | WetherVane",
  description:
    "All polling data tracked by WetherVane for 2026 Senate, Governor, and special election races.",
  openGraph: {
    title: "2026 Election Polls | WetherVane",
    description:
      "Browse all polls tracked by WetherVane — sortable by date, race, pollster, and grade for 2026 Senate, Governor, and special elections.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "2026 Election Polls | WetherVane",
    description:
      "All polling data tracked by WetherVane for 2026 Senate, Governor, and special election races.",
  },
};

// ── JSON-LD ────────────────────────────────────────────────────────────────

const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "WebPage",
  name: "2026 Election Polls — WetherVane",
  description:
    "All polling data tracked by WetherVane for 2026 Senate, Governor, and special election races. Sortable by date, race, pollster, sample size, and grade.",
  url: "https://wethervane.hhaines.duckdns.org/polls",
  isPartOf: {
    "@type": "WebSite",
    name: "WetherVane",
    url: "https://wethervane.hhaines.duckdns.org",
  },
};

// ── Data fetch ─────────────────────────────────────────────────────────────

const API_BASE = process.env.API_URL || "http://localhost:8002";

async function fetchPolls(): Promise<PollEntry[] | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/polls`, {
      next: { revalidate: 3600 }, // hourly — polls update more often than accuracy data
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ── Page ───────────────────────────────────────────────────────────────────

export default async function PollsPage() {
  const polls = await fetchPolls();

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
      />

      <div className="max-w-5xl mx-auto py-8 px-4 pb-20">
        {/* Breadcrumb */}
        <nav
          aria-label="breadcrumb"
          className="text-xs mb-6"
          style={{ color: "var(--color-text-muted)" }}
        >
          <ol className="flex flex-wrap items-center gap-x-1 list-none p-0 m-0">
            <li>
              <Link
                href="/"
                style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
              >
                Home
              </Link>
            </li>
            <li aria-hidden="true">/</li>
            <li aria-current="page">Polls</li>
          </ol>
        </nav>

        {/* Header */}
        <header className="mb-8">
          <h1
            className="text-3xl font-bold mb-3"
            style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
          >
            2026 Election Polls
          </h1>
          <p
            className="text-sm leading-relaxed max-w-2xl"
            style={{ color: "var(--color-text-muted)" }}
          >
            All polling data tracked by WetherVane for 2026 Senate, Governor,
            and special election races. Polls are collected from public sources
            including Wikipedia, 270toWin, and RealClearPolitics.
            {polls && (
              <span> {polls.length} polls across {new Set(polls.map((p) => p.race)).size} races.</span>
            )}
          </p>
        </header>

        {/* Table */}
        {polls ? (
          <PollsTable polls={polls} />
        ) : (
          <div
            className="text-center py-16 rounded-md"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            <p>Could not load poll data. Please try again later.</p>
          </div>
        )}
      </div>
    </>
  );
}
