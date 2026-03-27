/**
 * Embed page — a compact, iframe-embeddable forecast card for a single race.
 *
 * Designed for bloggers and journalists:
 *   <iframe src="https://wethervane.hhaines.duckdns.org/embed/2026-fl-senate"
 *           width="400" height="200" frameborder="0"></iframe>
 *
 * The parent embed/layout.tsx provides a clean HTML shell with no global
 * WetherVane chrome (no nav, no map, no overflow:hidden body).
 */
import type { Metadata } from "next";

// ── Types ──────────────────────────────────────────────────────────────────

interface RaceDetail {
  race: string;
  slug: string;
  state_abbr: string;
  race_type: string;
  year: number;
  prediction: number | null;
  n_counties: number;
}

// ── Constants ──────────────────────────────────────────────────────────────

// Server-side fetch goes directly to the API; client never calls this URL.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

const DEM_COLOR = "#2166ac";
const REP_COLOR = "#d73027";
const MUTED_COLOR = "#666666";
const BORDER_COLOR = "#e0e0e0";

// ── Helpers ────────────────────────────────────────────────────────────────

async function fetchRaceDetail(slug: string): Promise<RaceDetail | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race/${slug}`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function fetchRaceSlugs(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race-slugs`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

/**
 * Return the Dem two-party vote share formatted as a partisan lean label and
 * the corresponding hex color.
 */
function formatLean(demShare: number | null): { label: string; color: string } {
  if (demShare === null) return { label: "No prediction", color: MUTED_COLOR };
  const margin = Math.abs(demShare - 0.5) * 100;
  if (demShare > 0.5) {
    return { label: `D+${margin.toFixed(1)}`, color: DEM_COLOR };
  }
  return { label: `R+${margin.toFixed(1)}`, color: REP_COLOR };
}

/** Format a 0–1 fraction as a percentage string, e.g. "52.3%". */
function formatPct(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

// ── Metadata ───────────────────────────────────────────────────────────────

type PageProps = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const data = await fetchRaceDetail(slug);
  const title = data
    ? `${data.year} ${data.state_abbr} ${data.race_type} — WetherVane`
    : "Race Forecast — WetherVane";
  return {
    title,
    robots: { index: false, follow: false },
  };
}

// ── Static params (pre-render all known races) ─────────────────────────────

export async function generateStaticParams() {
  const slugs = await fetchRaceSlugs();
  return slugs.map((slug) => ({ slug }));
}

// ── Page component ─────────────────────────────────────────────────────────

export default async function EmbedPage({ params }: PageProps) {
  const { slug } = await params;
  const data = await fetchRaceDetail(slug);

  // ── Error state ──────────────────────────────────────────────────────────
  if (!data) {
    return (
      <div style={{
        maxWidth: 400,
        margin: "0 auto",
        padding: "20px 16px",
        textAlign: "center",
        color: MUTED_COLOR,
        fontSize: 14,
      }}>
        <p style={{ margin: "0 0 8px" }}>Race not found.</p>
        <a href="https://wethervane.hhaines.duckdns.org/forecast"
           style={{ color: DEM_COLOR, fontSize: 12 }}>
          View all forecasts
        </a>
      </div>
    );
  }

  const lean = formatLean(data.prediction);
  const demPct = data.prediction !== null ? formatPct(data.prediction) : "—";
  const repPct = data.prediction !== null ? formatPct(1 - data.prediction) : "—";

  // Win probability: a naive threshold-based label derived from the lean
  // magnitude.  Full probabilistic intervals require the pred_std field which
  // is not part of RaceDetail; this gives bloggers a quick directional read.
  const winProb: string = (() => {
    if (data.prediction === null) return "—";
    const margin = Math.abs(data.prediction - 0.5) * 100;
    // Rough mapping: margin → probability label
    if (margin >= 15) return data.prediction > 0.5 ? "Dem likely" : "Rep likely";
    if (margin >= 7) return data.prediction > 0.5 ? "Lean Dem" : "Lean Rep";
    if (margin >= 3) return data.prediction > 0.5 ? "Slight Dem" : "Slight Rep";
    return "Toss-up";
  })();

  const raceTitle = `${data.year} ${data.state_abbr} ${data.race_type}`;
  const raceUrl = `https://wethervane.hhaines.duckdns.org/forecast/${slug}`;

  // Bar widths normalise automatically since dem+rep always sum to 1.
  const demFrac = data.prediction ?? 0.5;
  const repFrac = 1 - demFrac;

  return (
    <div style={{
      maxWidth: 400,
      margin: "0 auto",
      padding: "14px 16px 10px",
      background: "#ffffff",
      border: `1px solid ${BORDER_COLOR}`,
      borderRadius: 6,
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }}>
      {/* Race title */}
      <div style={{ marginBottom: 10 }}>
        <a href={raceUrl} target="_blank" rel="noopener noreferrer"
           style={{
             fontSize: 15,
             fontWeight: 700,
             color: "#222222",
             fontFamily: "Georgia, 'Times New Roman', serif",
             lineHeight: 1.2,
           }}>
          {raceTitle}
        </a>
      </div>

      {/* Lean badge + win label */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{
          display: "inline-block",
          padding: "3px 10px",
          borderRadius: 4,
          fontSize: 14,
          fontWeight: 700,
          color: lean.color,
          border: `1px solid ${lean.color}`,
          background: "#f7f8fa",
        }}>
          {lean.label}
        </span>
        <span style={{ fontSize: 13, color: MUTED_COLOR }}>{winProb}</span>
      </div>

      {/* D / R share bars */}
      <div style={{ marginBottom: 10 }}>
        {/* Dem row */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
          <span style={{
            width: 28,
            fontSize: 12,
            fontWeight: 700,
            color: DEM_COLOR,
            textAlign: "right",
            flexShrink: 0,
          }}>D</span>
          <div style={{
            flex: 1,
            height: 14,
            background: "#e8f0f7",
            borderRadius: 3,
            overflow: "hidden",
          }}>
            <div style={{
              width: `${demFrac * 100}%`,
              height: "100%",
              background: DEM_COLOR,
              borderRadius: 3,
            }} />
          </div>
          <span style={{
            width: 42,
            fontSize: 13,
            fontWeight: 600,
            color: DEM_COLOR,
            flexShrink: 0,
          }}>{demPct}</span>
        </div>

        {/* Rep row */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            width: 28,
            fontSize: 12,
            fontWeight: 700,
            color: REP_COLOR,
            textAlign: "right",
            flexShrink: 0,
          }}>R</span>
          <div style={{
            flex: 1,
            height: 14,
            background: "#fce8e6",
            borderRadius: 3,
            overflow: "hidden",
          }}>
            <div style={{
              width: `${repFrac * 100}%`,
              height: "100%",
              background: REP_COLOR,
              borderRadius: 3,
            }} />
          </div>
          <span style={{
            width: 42,
            fontSize: 13,
            fontWeight: 600,
            color: REP_COLOR,
            flexShrink: 0,
          }}>{repPct}</span>
        </div>
      </div>

      {/* Footer: county count + branding */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        paddingTop: 8,
        borderTop: `1px solid ${BORDER_COLOR}`,
        fontSize: 11,
        color: MUTED_COLOR,
      }}>
        <span>
          {data.n_counties} {data.n_counties === 1 ? "county" : "counties"} in model
        </span>
        <a href="https://wethervane.hhaines.duckdns.org"
           target="_blank"
           rel="noopener noreferrer"
           style={{ color: MUTED_COLOR, fontWeight: 600 }}>
          Powered by WetherVane
        </a>
      </div>
    </div>
  );
}
