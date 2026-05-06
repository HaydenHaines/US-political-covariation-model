"use client";

import Link from "next/link";
import { useSenateOverview } from "@/lib/hooks/use-senate-overview";
import { useGovernorOverview } from "@/lib/hooks/use-governor-overview";
import { PALETTE, DUSTY_INK } from "@/lib/config/palette";
import { ELECTION_YEAR } from "@/lib/config/election";
import { Skeleton } from "@/components/ui/skeleton";

const DEM = PALETTE.DEM_PRIMARY;
const GOP = PALETTE.GOP_PRIMARY;
const TOSSUP = DUSTY_INK.tossup;

const D_RATINGS = new Set(["safe_d", "likely_d", "lean_d"]);
const R_RATINGS = new Set(["safe_r", "likely_r", "lean_r"]);

interface PartisanBarProps {
  dem: number;
  gop: number;
  tossup: number;
  total: number;
  majorityLine?: number;
}

function PartisanBar({ dem, gop, tossup, total, majorityLine }: PartisanBarProps) {
  const demPct = (dem / total) * 100;
  const tossupPct = (tossup / total) * 100;
  const gopPct = (gop / total) * 100;
  const majorityPct = majorityLine ? (majorityLine / total) * 100 : null;

  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          display: "flex",
          height: 7,
          borderRadius: 4,
          overflow: "hidden",
          gap: 1,
        }}
      >
        <div style={{ width: `${demPct}%`, background: DEM, borderRadius: "4px 0 0 4px" }} />
        {tossup > 0 && (
          <div style={{ width: `${tossupPct}%`, background: TOSSUP }} />
        )}
        <div style={{ width: `${gopPct}%`, background: GOP, borderRadius: "0 4px 4px 0" }} />
      </div>
      {majorityPct !== null && (
        <div
          style={{
            position: "absolute",
            top: -4,
            left: `${majorityPct}%`,
            width: 2,
            height: 15,
            background: "var(--color-text)",
            borderRadius: 1,
            opacity: 0.4,
          }}
        />
      )}
    </div>
  );
}

interface SeatCountProps {
  dem: number;
  gop: number;
  unit: string;
}

function SeatCount({ dem, gop, unit }: SeatCountProps) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
      <span
        style={{
          fontSize: 26,
          fontWeight: 700,
          fontFamily: "Georgia, 'Times New Roman', serif",
          color: DEM,
          lineHeight: 1,
        }}
      >
        {dem}
      </span>
      <span
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.05em",
          color: DEM,
          opacity: 0.75,
        }}
      >
        D
      </span>
      <span
        style={{
          fontSize: 13,
          color: "var(--color-border)",
          margin: "0 2px",
        }}
      >
        /
      </span>
      <span
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.05em",
          color: GOP,
          opacity: 0.75,
        }}
      >
        R
      </span>
      <span
        style={{
          fontSize: 26,
          fontWeight: 700,
          fontFamily: "Georgia, 'Times New Roman', serif",
          color: GOP,
          lineHeight: 1,
        }}
      >
        {gop}
      </span>
      <span
        style={{
          fontSize: 11,
          color: "var(--color-text-muted)",
          marginLeft: 2,
        }}
      >
        {unit}
      </span>
    </div>
  );
}

export function NationalProjectionStrip() {
  const senate = useSenateOverview();
  const governor = useGovernorOverview();

  const loading = senate.isLoading || governor.isLoading;
  const error = senate.error || governor.error;

  if (loading) {
    return (
      <div style={{ paddingBottom: 8 }}>
        <Skeleton className="h-4 w-48 mb-5" />
        <div style={{ display: "flex", gap: 16 }}>
          <div style={{ flex: 1 }}>
            <Skeleton className="h-3 w-16 mb-3" />
            <Skeleton className="h-8 w-40 mb-2" />
            <Skeleton className="h-2 w-full" />
          </div>
          <div style={{ width: 1, background: "var(--color-border)" }} />
          <div style={{ flex: 1 }}>
            <Skeleton className="h-3 w-20 mb-3" />
            <Skeleton className="h-8 w-36 mb-2" />
            <Skeleton className="h-2 w-full" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !senate.data || !governor.data) {
    return null;
  }

  const { dem_projected, gop_projected } = senate.data;
  const senateTossup = 100 - dem_projected - gop_projected;

  const govRaces = governor.data.races;
  const demGov = govRaces.filter((r) => D_RATINGS.has(r.rating)).length;
  const gopGov = govRaces.filter((r) => R_RATINGS.has(r.rating)).length;
  const tossupGov = govRaces.filter((r) => r.rating === "tossup").length;
  const govTotal = govRaces.length;

  return (
    <div style={{ marginBottom: 24, paddingBottom: 20, borderBottom: "1px solid var(--color-border)" }}>
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--color-text-subtle)",
          marginBottom: 14,
        }}
      >
        {ELECTION_YEAR} Projected Outcome
      </div>

      <div style={{ display: "flex", gap: 0 }}>
        {/* Senate block */}
        <div style={{ flex: 1, minWidth: 0, paddingRight: 16 }}>
          <Link
            href="/forecast/senate"
            style={{ textDecoration: "none" }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "var(--color-text-muted)",
                letterSpacing: "0.05em",
                marginBottom: 8,
              }}
            >
              Senate
            </div>
            <SeatCount dem={dem_projected} gop={gop_projected} unit="seats" />
            <div style={{ marginTop: 8, marginBottom: 6 }}>
              <PartisanBar
                dem={dem_projected}
                gop={gop_projected}
                tossup={senateTossup}
                total={100}
                majorityLine={51}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 10, color: "var(--color-text-subtle)" }}>
                51 for control
              </span>
              {senateTossup > 0 && (
                <span style={{ fontSize: 10, color: TOSSUP, opacity: 0.9 }}>
                  {senateTossup} tossup
                </span>
              )}
            </div>
          </Link>
        </div>

        {/* Divider */}
        <div
          style={{
            width: 1,
            background: "var(--color-border)",
            margin: "4px 0",
            flexShrink: 0,
          }}
        />

        {/* Governor block */}
        <div style={{ flex: 1, minWidth: 0, paddingLeft: 16 }}>
          <Link
            href="/forecast/governor"
            style={{ textDecoration: "none" }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "var(--color-text-muted)",
                letterSpacing: "0.05em",
                marginBottom: 8,
              }}
            >
              Governors
            </div>
            <SeatCount dem={demGov} gop={gopGov} unit="wins" />
            <div style={{ marginTop: 8, marginBottom: 6 }}>
              <PartisanBar
                dem={demGov}
                gop={gopGov}
                tossup={tossupGov}
                total={govTotal}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 10, color: "var(--color-text-subtle)" }}>
                {govTotal} races
              </span>
              {tossupGov > 0 && (
                <span style={{ fontSize: 10, color: TOSSUP, opacity: 0.9 }}>
                  {tossupGov} tossup
                </span>
              )}
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
