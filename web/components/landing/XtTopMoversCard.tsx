"use client";

import Link from "next/link";
import useSWR from "swr";
import type { XtImpactResponse } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

async function fetchTop5(raceType?: string): Promise<XtImpactResponse> {
  const url = `/api/forecast/xt-impact?limit=5${raceType ? `&race_type=${encodeURIComponent(raceType)}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`xt-impact failed: ${res.status}`);
  return res.json();
}

function formatRaceLabel(raceId: string): string {
  return raceId
    .split("-")
    .filter((p) => !/^\d{4}$/.test(p))
    .map((p) => (p.length <= 2 ? p.toUpperCase() : p.charAt(0).toUpperCase() + p.slice(1)))
    .join(" ");
}

function DeltaBar({ delta, maxAbs }: { delta: number; maxAbs: number }) {
  const pct = maxAbs > 0 ? (Math.abs(delta) / maxAbs) * 50 : 4;
  const isDem = delta > 0;

  return (
    <div
      aria-hidden="true"
      style={{
        position: "relative",
        width: "100%",
        height: 6,
        background: "var(--color-border)",
        borderRadius: 3,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          ...(isDem
            ? { left: "50%", width: `${pct}%` }
            : { right: "50%", width: `${pct}%` }),
          background: isDem ? "var(--color-dem)" : "var(--color-rep)",
          borderRadius: 3,
        }}
      />
    </div>
  );
}

interface XtTopMoversCardProps {
  raceType?: string;
}

export function XtTopMoversCard({ raceType }: XtTopMoversCardProps = {}) {
  const swrKey = raceType ? `xt-impact-5-${raceType}` : "xt-impact-5";
  const { data, isLoading } = useSWR<XtImpactResponse>(
    swrKey,
    () => fetchTop5(raceType),
    { revalidateOnFocus: false, dedupingInterval: 3_600_000 },
  );

  const cardStyle: React.CSSProperties = {
    border: "1px solid var(--color-border)",
    borderRadius: 8,
    background: "var(--color-surface)",
    padding: "16px 20px",
  };

  if (isLoading || !data) {
    return (
      <div style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
          <Skeleton className="h-3 w-36" />
          <Skeleton className="h-3 w-24" />
        </div>
        {[...Array(5)].map((_, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-2 flex-1" />
            <Skeleton className="h-4 w-14" />
            <Skeleton className="h-4 w-8" />
          </div>
        ))}
      </div>
    );
  }

  const movers = data.top_movers.slice(0, 5);
  const maxAbs = Math.max(...movers.map((m) => Math.abs(m.delta_pp)), 0.01);

  return (
    <div style={cardStyle}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 14,
        }}
      >
        <h3
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: "var(--color-text)",
            margin: 0,
            textTransform: "uppercase",
            letterSpacing: "0.07em",
          }}
        >
          {raceType
            ? `Top ${raceType.charAt(0).toUpperCase() + raceType.slice(1)} Movers`
            : "Top Senate Movers"}
        </h3>
        <span style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
          Cross-type poll impact
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {movers.map((mover) => {
          const isDem = mover.delta_pp > 0;
          return (
            <div
              key={mover.race_id}
              style={{ display: "flex", alignItems: "center", gap: 10 }}
            >
              <Link
                href={`/forecast/${mover.race_id}`}
                style={{
                  width: 76,
                  flexShrink: 0,
                  fontSize: 12,
                  fontWeight: 500,
                  color: "inherit",
                  textDecoration: "none",
                }}
                className="hover:underline"
              >
                {formatRaceLabel(mover.race_id)}
              </Link>

              <div style={{ flex: 1, minWidth: 60 }}>
                <DeltaBar delta={mover.delta_pp} maxAbs={maxAbs} />
              </div>

              <span
                style={{
                  width: 52,
                  textAlign: "right",
                  fontSize: 12,
                  fontFamily: "var(--font-mono, monospace)",
                  fontWeight: 600,
                  flexShrink: 0,
                  color: isDem ? "var(--color-dem)" : "var(--color-rep)",
                }}
              >
                {isDem ? "+" : ""}
                {mover.delta_pp.toFixed(1)} pp
              </span>

              <span
                style={{
                  flexShrink: 0,
                  fontSize: 10,
                  fontFamily: "var(--font-mono, monospace)",
                  color: "var(--color-text-muted)",
                  background: "var(--color-border)",
                  borderRadius: 10,
                  padding: "1px 6px",
                  whiteSpace: "nowrap",
                }}
              >
                {mover.n_xt_polls}×
              </span>
            </div>
          );
        })}
      </div>

      {data.report_date && (
        <p
          style={{
            marginTop: 12,
            fontSize: 10,
            color: "var(--color-text-muted)",
            textAlign: "right",
          }}
        >
          As of {data.report_date}
        </p>
      )}
    </div>
  );
}
