import type { Metadata } from "next";
import Link from "next/link";
import { PollsterTable } from "./PollsterTable";

// ── Metadata ───────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "Pollster Accuracy | WetherVane",
  description:
    "Ranked accuracy metrics for political pollsters based on 2022 election backtesting — RMSE and mean bias in percentage points.",
  openGraph: {
    title: "Pollster Accuracy | WetherVane",
    description:
      "How accurate were the pollsters? 2022 election backtest results ranked by RMSE for every pollster in the WetherVane dataset.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "Pollster Accuracy | WetherVane",
    description:
      "2022 election backtest RMSE and bias rankings for political pollsters — from WetherVane's structural forecast model.",
  },
};

// ── JSON-LD ────────────────────────────────────────────────────────────────

const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "WebPage",
  name: "Pollster Accuracy — WetherVane 2022 Backtest Rankings",
  description:
    "Ranked accuracy metrics for political pollsters based on 2022 general election backtesting. RMSE and mean signed error in percentage points of two-party vote share.",
  url: "https://wethervane.hhaines.duckdns.org/pollsters",
  isPartOf: {
    "@type": "WebSite",
    name: "WetherVane",
    url: "https://wethervane.hhaines.duckdns.org",
  },
};

// ── Data fetch ─────────────────────────────────────────────────────────────

const API_BASE = process.env.API_URL || "http://localhost:8002";

interface PollsterEntry {
  pollster: string;
  rank: number;
  n_polls: number;
  n_races: number;
  rmse_pp: number;
  mean_error_pp: number;
}

interface PollsterAccuracyResponse {
  description: string;
  n_pollsters: number;
  pollsters: PollsterEntry[];
}

async function fetchPollsterAccuracy(): Promise<PollsterAccuracyResponse | null> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/pollsters/accuracy`, {
      next: { revalidate: 86400 }, // refresh once per day — data only changes when regenerated
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ── Page ───────────────────────────────────────────────────────────────────

export default async function PollstersPage() {
  const data = await fetchPollsterAccuracy();

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
            <li aria-current="page">Pollster Accuracy</li>
          </ol>
        </nav>

        {/* Header */}
        <header className="mb-8">
          <h1
            className="text-3xl font-bold mb-3"
            style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
          >
            Pollster Accuracy
          </h1>
          <p
            className="text-sm leading-relaxed max-w-2xl"
            style={{ color: "var(--color-text-muted)" }}
          >
            Ranked accuracy for political pollsters based on 2022 general
            election backtesting. Each pollster is measured against actual
            results using root mean squared error (RMSE) and mean signed
            error — both in percentage points of two-party vote share.
            Rank&nbsp;1 is the most accurate.
            {data?.n_pollsters && (
              <span> {data.n_pollsters} pollsters evaluated.</span>
            )}
          </p>

          {/* Methodology note */}
          <div
            className="mt-4 text-xs rounded-md px-4 py-3"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            <strong style={{ color: "var(--color-text)" }}>
              How accuracy is measured:{" "}
            </strong>
            RMSE (root mean squared error) penalizes large misses more heavily —
            lower is better. Mean bias shows systematic lean: positive values
            mean the pollster over-predicted Democratic share; negative means
            over-predicting Republican share.
          </div>
        </header>

        {/* Table */}
        {data ? (
          <PollsterTable
            pollsters={data.pollsters}
            description={data.description}
          />
        ) : (
          <div
            className="text-center py-16 rounded-md"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            <p>Could not load pollster accuracy data. Please try again later.</p>
          </div>
        )}
      </div>
    </>
  );
}
