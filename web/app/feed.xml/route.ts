// RSS 2.0 feed for WetherVane forecast prediction changes.
// Each item = one scrape date where at least one tracked race moved by >1pp.
// Regenerated hourly via ISR revalidation.

export const revalidate = 3600;

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ||
  "https://wethervane.hhaines.duckdns.org";

// Server-side fetch goes direct to the API, not through the /api/* Caddy proxy.
const API_BASE = process.env.API_URL || "http://localhost:8002";

// Only report changes that are >= 1 percentage point (0.01) for RSS items.
// The API already filters at _MIN_CHANGELOG_DELTA (0.005), so we apply an
// additional threshold here to keep feed entries meaningful to subscribers.
const RSS_DELTA_THRESHOLD = 0.01;

// ---------------------------------------------------------------------------
// Types matching the API changelog response shape
// ---------------------------------------------------------------------------

interface RaceDiff {
  race: string;
  before: number | null;
  after: number | null;
  delta: number | null;
}

interface ChangelogEntry {
  date: string;
  note: string | null;
  diffs: RaceDiff[];
}

interface ChangelogResponse {
  entries: ChangelogEntry[];
  current_snapshot_date: string | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Convert "2026 FL Senate" → URL slug "2026-fl-senate". */
function raceToSlug(race: string): string {
  return race.toLowerCase().replace(/\s+/g, "-");
}

/** Format a dem-share fraction as a readable percentage, e.g. 0.523 → "52.3%". */
function fmtPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Build a human-readable summary of the diffs for one changelog entry.
 * Shows direction (D+/R+) and magnitude for each changed race.
 */
function buildItemDescription(
  entry: ChangelogEntry,
  significantDiffs: RaceDiff[],
): string {
  const lines: string[] = [];

  if (entry.note) {
    lines.push(entry.note);
    lines.push("");
  }

  lines.push(`${significantDiffs.length} race(s) moved by ≥1pp on ${entry.date}:`);
  lines.push("");

  for (const d of significantDiffs) {
    if (d.before === null || d.after === null || d.delta === null) {
      // Newly tracked race
      const label = d.after !== null ? `${fmtPct(d.after)} Dem` : "unknown";
      lines.push(`• ${d.race}: New race — ${label}`);
    } else {
      const direction = d.delta > 0 ? "D" : "R";
      const magnitude = Math.abs(d.delta * 100).toFixed(1);
      const slug = raceToSlug(d.race);
      const url = `${SITE_URL}/forecast/${slug}`;
      lines.push(
        `• ${d.race}: ${fmtPct(d.before)} → ${fmtPct(d.after)} (${direction}+${magnitude}pp) — ${url}`,
      );
    }
  }

  return lines.join("\n");
}

/**
 * Parse a YYYY-MM-DD date string into an RFC 2822 string for RSS pubDate.
 * Falls back to the current time if parsing fails.
 */
function toRssDate(dateStr: string): string {
  const parsed = new Date(dateStr + "T12:00:00Z");
  return isNaN(parsed.getTime()) ? new Date().toUTCString() : parsed.toUTCString();
}

// ---------------------------------------------------------------------------
// XML construction
// ---------------------------------------------------------------------------

function buildRssItem(
  entry: ChangelogEntry,
  significantDiffs: RaceDiff[],
): string {
  const pubDate = toRssDate(entry.date);
  const changedCount = significantDiffs.length;

  // Title: "Apr 3, 2026 — 3 race(s) updated"
  const dateLabel = new Date(entry.date + "T12:00:00Z").toLocaleDateString(
    "en-US",
    { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" },
  );
  const title = `${dateLabel} — ${changedCount} race${changedCount === 1 ? "" : "s"} updated`;

  // Link: changelog page, anchored to the date
  const link = `${SITE_URL}/changelog#${entry.date}`;

  const description = escapeXml(buildItemDescription(entry, significantDiffs));

  return [
    "    <item>",
    `      <title>${escapeXml(title)}</title>`,
    `      <link>${link}</link>`,
    `      <guid isPermaLink="true">${link}</guid>`,
    `      <description>${description}</description>`,
    `      <pubDate>${pubDate}</pubDate>`,
    "    </item>",
  ].join("\n");
}

function buildFeedXml(entries: ChangelogEntry[], lastBuildDate: string): string {
  // Only include entries where at least one race moved >= RSS_DELTA_THRESHOLD.
  // Skip the initial baseline entry (diffs have null delta) since it has no
  // "change" to report — it's just the starting state.
  const items: string[] = [];

  for (const entry of entries) {
    const significantDiffs = entry.diffs.filter(
      (d) => d.delta !== null && Math.abs(d.delta) >= RSS_DELTA_THRESHOLD,
    );
    if (significantDiffs.length === 0) continue;
    items.push(buildRssItem(entry, significantDiffs));
  }

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
    "  <channel>",
    "    <title>WetherVane — Forecast Updates</title>",
    `    <link>${SITE_URL}</link>`,
    "    <description>Weekly updates when WetherVane's 2026 election forecasts change with new polling data. Each item lists which races moved and by how much.</description>",
    "    <language>en-us</language>",
    `    <lastBuildDate>${lastBuildDate}</lastBuildDate>`,
    `    <atom:link href="${SITE_URL}/feed.xml" rel="self" type="application/rss+xml" />`,
    ...items,
    "  </channel>",
    "</rss>",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

export async function GET(): Promise<Response> {
  const lastBuildDate = new Date().toUTCString();

  let changelog: ChangelogResponse = { entries: [], current_snapshot_date: null };

  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/changelog`, {
      next: { revalidate },
    });
    if (res.ok) {
      changelog = await res.json();
    }
  } catch {
    // API unavailable — return an empty but valid feed rather than 500.
  }

  const xml = buildFeedXml(changelog.entries, lastBuildDate);

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
    },
  });
}
