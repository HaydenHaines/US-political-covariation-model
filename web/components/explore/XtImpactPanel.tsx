"use client";

import Link from "next/link";
import useSWR from "swr";
import type { XtImpactResponse } from "@/lib/api";

async function fetchXtImpact(): Promise<XtImpactResponse> {
  const res = await fetch("/api/forecast/xt-impact?limit=10");
  if (!res.ok) throw new Error(`/api/forecast/xt-impact failed: ${res.status}`);
  return res.json();
}

function DeltaArrow({ delta }: { delta: number }) {
  if (Math.abs(delta) < 0.01) {
    return <span style={{ color: "var(--color-text-muted)" }}>—</span>;
  }
  const isDem = delta > 0;
  return (
    <span
      aria-label={isDem ? "Dem gain" : "GOP gain"}
      style={{ color: isDem ? "var(--color-dem)" : "var(--color-rep)" }}
    >
      {isDem ? "▲" : "▼"}
    </span>
  );
}

/** Formats a race_id like "2026-ga-senate" → "2026 GA Senate". */
function formatRaceId(raceId: string): string {
  return raceId
    .split("-")
    .map((part) =>
      /^\d{4}$/.test(part) ? part : part.toUpperCase().slice(0, 1) + part.slice(1),
    )
    .join(" ");
}

export function XtImpactPanel() {
  const { data, isLoading, error } = useSWR<XtImpactResponse>(
    "xt-impact",
    fetchXtImpact,
    {
      revalidateOnFocus: false,
      dedupingInterval: 3_600_000, // 1 hour — cached on backend too
    },
  );

  if (isLoading) {
    return (
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 8,
          padding: "24px",
          background: "var(--color-surface)",
          color: "var(--color-text-muted)",
          fontSize: 13,
        }}
      >
        Loading cross-type movers…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 8,
          padding: "24px",
          background: "var(--color-surface)",
          color: "var(--color-text-muted)",
          fontSize: 13,
        }}
      >
        Cross-type impact data unavailable.
      </div>
    );
  }

  const { top_movers, races_with_xt, report_date } = data;

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        background: "var(--color-surface)",
        overflowX: "auto",
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 13,
        }}
      >
        <thead>
          <tr>
            <th
              style={{
                padding: "12px 16px",
                textAlign: "left",
                fontWeight: 600,
                borderBottom: "2px solid var(--color-border)",
                color: "var(--color-text)",
              }}
            >
              Race
            </th>
            <th
              style={{
                padding: "12px 16px",
                textAlign: "right",
                fontWeight: 600,
                borderBottom: "2px solid var(--color-border)",
                color: "var(--color-text)",
                whiteSpace: "nowrap",
              }}
            >
              Cross-Type Δ (pp)
            </th>
            <th
              style={{
                padding: "12px 16px",
                textAlign: "right",
                fontWeight: 600,
                borderBottom: "2px solid var(--color-border)",
                color: "var(--color-text)",
                whiteSpace: "nowrap",
              }}
            >
              Polls
            </th>
            <th
              style={{
                padding: "12px 16px",
                textAlign: "center",
                fontWeight: 600,
                borderBottom: "2px solid var(--color-border)",
                color: "var(--color-text)",
              }}
            >
              Direction
            </th>
          </tr>
        </thead>
        <tbody>
          {top_movers.map((mover, i) => (
            <tr
              key={mover.race_id}
              style={{
                background: i % 2 === 0 ? "transparent" : "rgba(0,0,0,0.015)",
              }}
            >
              <td
                style={{
                  padding: "8px 16px",
                  borderBottom: "1px solid var(--color-border)",
                  color: "var(--color-text)",
                  fontFamily: "var(--font-sans)",
                }}
              >
                <Link
                  href={`/forecast/${mover.race_id}`}
                  style={{ color: "inherit", textDecoration: "none" }}
                  className="hover:underline"
                >
                  {formatRaceId(mover.race_id)}
                </Link>
              </td>
              <td
                style={{
                  padding: "8px 16px",
                  borderBottom: "1px solid var(--color-border)",
                  textAlign: "right",
                  fontFamily: "var(--font-mono, monospace)",
                  color: mover.delta_pp > 0
                    ? "var(--color-dem)"
                    : mover.delta_pp < 0
                    ? "var(--color-rep)"
                    : "var(--color-text-muted)",
                  fontWeight: 600,
                }}
              >
                {mover.delta_pp > 0 ? "+" : ""}
                {mover.delta_pp.toFixed(2)}
              </td>
              <td
                style={{
                  padding: "8px 16px",
                  borderBottom: "1px solid var(--color-border)",
                  textAlign: "right",
                  fontFamily: "var(--font-mono, monospace)",
                  color: "var(--color-text-muted)",
                }}
              >
                {mover.n_xt_polls}
              </td>
              <td
                style={{
                  padding: "8px 16px",
                  borderBottom: "1px solid var(--color-border)",
                  textAlign: "center",
                }}
              >
                <DeltaArrow delta={mover.delta_pp} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div
        style={{
          padding: "10px 16px",
          fontSize: 11,
          color: "var(--color-text-muted)",
          borderTop: "1px solid var(--color-border)",
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 4,
        }}
      >
        <span>{races_with_xt} races with cross-type poll data</span>
        {report_date && <span>As of {report_date}</span>}
      </div>
    </div>
  );
}
