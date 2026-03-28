"use client";
import { useState, useEffect, useCallback } from "react";
import { useMapContext } from "@/components/MapContext";
import { fetchPolls, feedMultiplePolls, type PollRow, type ForecastRow } from "@/lib/api";
import { DUSTY_INK } from "@/lib/colors";
import { formatMargin, marginColor } from "@/lib/typeDisplay";

const YEAR = "2026";
const PANEL_WIDTH = 340;
const MAX_POLLS = 5;

/** Build a type_id -> dem_share choropleth map from forecast rows. */
function buildChoropleth(rows: ForecastRow[], stateFilter?: string): Map<string, number> {
  const filtered = stateFilter
    ? rows.filter((r) => r.state_abbr === stateFilter)
    : rows;
  const typeSum = new Map<number, number>();
  const typeCount = new Map<number, number>();
  filtered.forEach((r) => {
    const typeId = (r as any).dominant_type;
    if (typeId != null && r.pred_dem_share != null) {
      typeSum.set(typeId, (typeSum.get(typeId) ?? 0) + r.pred_dem_share);
      typeCount.set(typeId, (typeCount.get(typeId) ?? 0) + 1);
    }
  });
  const m = new Map<string, number>();
  typeSum.forEach((sum, typeId) => {
    m.set(String(typeId), sum / (typeCount.get(typeId) ?? 1));
  });
  return m;
}

/** Extract state abbreviation from a race string like "2026 GA Senate". */
function extractState(race: string): string | null {
  const m = race.match(/\d{4}\s+([A-Z]{2})\s+/);
  return m ? m[1] : null;
}

/** Sliding right panel shown in dashboard mode when a state is zoomed. */
export function DashboardOverlay() {
  const {
    zoomedState,
    layoutMode,
    forecastChoropleth,
    setForecastChoropleth,
  } = useMapContext();

  const [polls, setPolls] = useState<PollRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [raceName, setRaceName] = useState<string | null>(null);
  const [statePred, setStatePred] = useState<number | null>(null);

  const visible = layoutMode === "dashboard" && zoomedState !== null;

  // Fetch polls when zoomed state changes
  useEffect(() => {
    if (!visible || !zoomedState) return;
    fetchPolls({ state: zoomedState, cycle: YEAR })
      .then((rows) => {
        setPolls(rows);
        // Derive race name from first poll
        if (rows.length > 0) {
          setRaceName(rows[0].race);
        } else {
          setRaceName(null);
        }
      })
      .catch(() => setPolls([]));
  }, [visible, zoomedState]);

  const handleRecalculate = useCallback(async () => {
    if (!zoomedState) return;
    setLoading(true);
    try {
      const result = await feedMultiplePolls({
        cycle: YEAR,
        state: zoomedState,
        race: raceName ?? undefined,
      });
      const choro = buildChoropleth(result.counties, zoomedState);
      setForecastChoropleth(choro);
      // Compute vote-weighted state prediction
      const stateRows = result.counties.filter((r) => r.state_abbr === zoomedState);
      if (stateRows.length > 0 && stateRows[0].state_pred != null) {
        setStatePred(stateRows[0].state_pred);
      } else {
        // Simple average fallback
        const vals = stateRows.filter((r) => r.pred_dem_share != null).map((r) => r.pred_dem_share!);
        setStatePred(vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null);
      }
    } finally {
      setLoading(false);
    }
  }, [zoomedState, raceName, setForecastChoropleth]);

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        width: PANEL_WIDTH,
        height: "100%",
        background: DUSTY_INK.background,
        borderLeft: `1px solid ${DUSTY_INK.border}`,
        boxShadow: "-4px 0 16px rgba(0,0,0,0.08)",
        transform: visible ? "translateX(0)" : `translateX(${PANEL_WIDTH}px)`,
        transition: "transform 250ms ease-out",
        zIndex: 20,
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
        overflow: "hidden",
        pointerEvents: visible ? "auto" : "none",
      }}
      aria-hidden={!visible}
    >
      {/* Header */}
      <div style={{
        padding: "16px 20px 12px",
        borderBottom: `1px solid ${DUSTY_INK.border}`,
      }}>
        <h2 style={{
          margin: 0,
          fontSize: 18,
          fontWeight: 700,
          color: DUSTY_INK.text,
          fontFamily: "var(--font-serif, Georgia, serif)",
        }}>
          {zoomedState ?? ""}
          {raceName ? ` — ${raceName.replace(/^\d{4}\s+[A-Z]{2}\s+/, "")}` : ""}
        </h2>

        {/* Current prediction margin */}
        {statePred !== null && (
          <div style={{
            marginTop: 8,
            fontSize: 14,
            fontWeight: 600,
            color: marginColor(statePred),
          }}>
            {formatMargin(statePred)}
          </div>
        )}
        {forecastChoropleth && statePred === null && (
          <div style={{ marginTop: 8, fontSize: 12, color: DUSTY_INK.textMuted }}>
            Choropleth active
          </div>
        )}
      </div>

      {/* Poll list */}
      <div style={{ flex: 1, overflow: "auto", padding: "12px 20px" }}>
        <h3 style={{
          margin: "0 0 8px",
          fontSize: 12,
          fontWeight: 600,
          color: DUSTY_INK.textMuted,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}>
          Recent Polls
        </h3>
        {polls.length === 0 && (
          <div style={{ fontSize: 12, color: DUSTY_INK.textSubtle }}>
            No polls available for this state.
          </div>
        )}
        {polls.slice(0, MAX_POLLS).map((p, i) => (
          <div
            key={i}
            style={{
              padding: "8px 0",
              borderBottom: i < Math.min(polls.length, MAX_POLLS) - 1
                ? `1px solid ${DUSTY_INK.border}`
                : "none",
              fontSize: 12,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: DUSTY_INK.text, fontWeight: 500 }}>
                {p.pollster ?? "Unknown"}
              </span>
              <span style={{ color: marginColor(p.dem_share), fontWeight: 600 }}>
                {formatMargin(p.dem_share)}
              </span>
            </div>
            <div style={{ color: DUSTY_INK.textSubtle, fontSize: 11, marginTop: 2 }}>
              {p.date ?? "No date"} · n={p.n_sample}
            </div>
          </div>
        ))}
        {polls.length > MAX_POLLS && (
          <div style={{ fontSize: 11, color: DUSTY_INK.textSubtle, marginTop: 4 }}>
            +{polls.length - MAX_POLLS} more
          </div>
        )}
      </div>

      {/* Recalculate button */}
      <div style={{ padding: "12px 20px 16px" }}>
        <button
          onClick={handleRecalculate}
          disabled={loading || polls.length === 0}
          style={{
            width: "100%",
            padding: "10px 0",
            borderRadius: 6,
            border: "none",
            background: polls.length > 0 ? DUSTY_INK.text : DUSTY_INK.border,
            color: polls.length > 0 ? "#fff" : DUSTY_INK.textSubtle,
            fontSize: 13,
            fontWeight: 600,
            fontFamily: "var(--font-sans)",
            cursor: polls.length > 0 ? "pointer" : "not-allowed",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? "Recalculating..." : "Recalculate with Polls"}
        </button>
      </div>
    </div>
  );
}
