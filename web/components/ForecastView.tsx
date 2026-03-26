"use client";
import { useEffect, useMemo, useState, useCallback } from "react";
import { fetchForecast, type ForecastRow } from "@/lib/api";
import { useMapContext } from "@/components/MapContext";

const LEAN_LABELS = [
  { threshold: 0.55, label: "Solid D", color: "var(--color-dem)" },
  { threshold: 0.525, label: "Likely D", color: "#5b9bd5" },
  { threshold: 0.505, label: "Lean D", color: "#9ec7e8" },
  { threshold: 0.495, label: "Toss-up", color: "#888" },
  { threshold: 0.475, label: "Lean R", color: "#f5a89a" },
  { threshold: 0.45, label: "Likely R", color: "#e07065" },
  { threshold: 0, label: "Solid R", color: "var(--color-rep)" },
];

function getLean(share: number) {
  for (const l of LEAN_LABELS) {
    if (share >= l.threshold) return l;
  }
  return LEAN_LABELS[LEAN_LABELS.length - 1];
}

function SectionHeader({
  title,
  disabled,
  open,
  onToggle,
}: {
  title: string;
  disabled?: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      style={{
        width: "100%",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "8px 10px",
        background: disabled ? "#f5f5f5" : "white",
        border: "1px solid var(--color-border)",
        borderRadius: "3px",
        cursor: disabled ? "not-allowed" : "pointer",
        fontSize: "12px",
        fontFamily: "var(--font-sans)",
        color: disabled ? "var(--color-text-muted)" : "#333",
        marginBottom: "2px",
      }}
    >
      <span style={{ fontWeight: 600 }}>{title}</span>
      {disabled ? (
        <span style={{ fontSize: "10px", background: "#e8e8e8", padding: "2px 6px", borderRadius: "10px" }}>
          coming soon
        </span>
      ) : (
        <span>{open ? "▲" : "▼"}</span>
      )}
    </button>
  );
}

/** Extract 2-char state abbreviation from a race label.
 * Handles the actual API format: "2026 FL Senate", "2026 GA Governor".
 * Walks whitespace-delimited tokens and returns the first all-uppercase 2-char token.
 */
function stateFromRace(race: string): string | null {
  for (const part of race.trim().split(/\s+/)) {
    if (/^[A-Z]{2}$/.test(part)) return part;
  }
  return null;
}

export function ForecastView() {
  const { setForecastState, setForecastChoropleth } = useMapContext();

  const [allRows, setAllRows] = useState<ForecastRow[]>([]);
  const [displayRows, setDisplayRows] = useState<ForecastRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [modelPriorOpen, setModelPriorOpen] = useState(true);

  // Sequential control state
  const [selectedState, setSelectedState] = useState<string>("");
  // Year is hardcoded 2026 for now (single cycle)
  const YEAR = "2026";
  const [selectedRace, setSelectedRace] = useState<string>("");

  // Load all forecast rows once to populate dropdowns
  useEffect(() => {
    fetchForecast().then((rows) => {
      setAllRows(rows);
    });
  }, []);

  // Derive available states from race labels ("2026 FL Senate" → "FL")
  const availableStates = useMemo(() => {
    const stateSet = new Set<string>();
    allRows.forEach((r) => {
      const s = stateFromRace(r.race);
      if (s) stateSet.add(s);
    });
    return Array.from(stateSet).sort();
  }, [allRows]);

  // Auto-select first state when rows load
  useEffect(() => {
    if (availableStates.length > 0 && !selectedState) {
      setSelectedState(availableStates[0]);
    }
  }, [availableStates, selectedState]);

  // Races available for selected state ("2026 FL Senate" matches "FL")
  const racesForState = useMemo(() => {
    if (!selectedState) return [];
    return Array.from(new Set(
      allRows
        .filter((r) => stateFromRace(r.race) === selectedState)
        .map((r) => r.race)
    )).sort();
  }, [allRows, selectedState]);

  // Auto-select first race when state changes
  useEffect(() => {
    if (racesForState.length > 0) {
      setSelectedRace(racesForState[0]);
    } else {
      setSelectedRace("");
    }
  }, [racesForState]);

  // Sync forecast state to map context for highlighting + pan
  useEffect(() => {
    setForecastState(selectedState || null);
  }, [selectedState, setForecastState]);

  // Recalculate: fetch model-prior predictions and push to map choropleth
  const recalculate = useCallback(async () => {
    if (!selectedRace) return;
    setLoading(true);
    try {
      const rows = await fetchForecast(selectedRace);
      setDisplayRows(rows);
      // Build fips → dem_share map for map choropleth
      const choropleth = new Map<string, number>();
      rows.forEach((r) => {
        if (r.pred_dem_share !== null) choropleth.set(r.county_fips, r.pred_dem_share);
      });
      setForecastChoropleth(choropleth);
    } finally {
      setLoading(false);
    }
  }, [selectedRace, setForecastChoropleth]);

  // Auto-recalculate when race changes
  useEffect(() => {
    if (selectedRace) recalculate();
    else { setDisplayRows([]); setForecastChoropleth(null); }
  }, [selectedRace]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clean up choropleth when leaving
  useEffect(() => () => { setForecastChoropleth(null); setForecastState(null); }, []);

  const stateRows = displayRows.filter((r) => r.state_abbr === selectedState);
  const statePred =
    stateRows.length > 0
      ? stateRows.reduce((sum, r) => sum + (r.pred_dem_share ?? 0), 0) / stateRows.length
      : null;
  const lean = statePred !== null ? getLean(statePred) : null;

  const dropdownStyle: React.CSSProperties = {
    padding: "6px 10px",
    border: "1px solid var(--color-border)",
    borderRadius: "3px",
    fontSize: "13px",
    fontFamily: "var(--font-sans)",
    background: "white",
    width: "100%",
  };

  const labelStyle: React.CSSProperties = {
    fontSize: "11px",
    color: "var(--color-text-muted)",
    display: "block",
    marginBottom: "3px",
    textTransform: "uppercase",
    letterSpacing: "0.4px",
  };

  return (
    <div style={{ padding: "14px" }}>
      {/* ── Sequential Controls ────────────────────────────────────── */}
      <div style={{ marginBottom: "14px" }}>
        <div style={{ marginBottom: "8px" }}>
          <label style={labelStyle}>State</label>
          <select
            value={selectedState}
            onChange={(e) => setSelectedState(e.target.value)}
            style={dropdownStyle}
          >
            {availableStates.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: "8px" }}>
          <label style={labelStyle}>Year</label>
          <select value={YEAR} disabled style={{ ...dropdownStyle, color: "var(--color-text-muted)" }}>
            <option value={YEAR}>{YEAR}</option>
          </select>
        </div>

        <div>
          <label style={labelStyle}>Election</label>
          <select
            value={selectedRace}
            onChange={(e) => setSelectedRace(e.target.value)}
            style={dropdownStyle}
            disabled={racesForState.length === 0}
          >
            {racesForState.length === 0 ? (
              <option value="">—</option>
            ) : (
              racesForState.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))
            )}
          </select>
        </div>
      </div>

      {/* ── Data Panel ─────────────────────────────────────────────── */}
      <div style={{ marginBottom: "12px" }}>
        <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.4px", color: "var(--color-text-muted)" }}>
          Data inputs
        </p>

        <SectionHeader
          title="Model Prior"
          open={modelPriorOpen}
          onToggle={() => setModelPriorOpen((p) => !p)}
        />
        {modelPriorOpen && (
          <div style={{
            padding: "8px 10px",
            border: "1px solid var(--color-border)",
            borderTop: "none",
            borderRadius: "0 0 3px 3px",
            marginBottom: "4px",
            fontSize: "12px",
            color: "var(--color-text-muted)",
            lineHeight: "1.5",
          }}>
            Structural baseline from type covariance (Ridge+HGB ensemble, LOO r=0.671).
            Predictions reflect 2024 presidential Dem share prior propagated through community type structure.
          </div>
        )}

        <SectionHeader title="Fundamentals" disabled open={false} onToggle={() => {}} />
        <SectionHeader title="National Polls" disabled open={false} onToggle={() => {}} />
        <SectionHeader title="State Polls" disabled open={false} onToggle={() => {}} />
      </div>

      {/* ── Recalculate ─────────────────────────────────────────────── */}
      <button
        onClick={recalculate}
        disabled={loading || !selectedRace}
        style={{
          width: "100%",
          padding: "8px",
          background: loading ? "#aaa" : "#2166ac",
          color: "white",
          border: "none",
          borderRadius: "3px",
          fontSize: "13px",
          fontFamily: "var(--font-sans)",
          fontWeight: 600,
          cursor: loading || !selectedRace ? "not-allowed" : "pointer",
          marginBottom: "14px",
          letterSpacing: "0.3px",
        }}
      >
        {loading ? "Calculating…" : "Recalculate"}
      </button>

      {/* ── State Summary ───────────────────────────────────────────── */}
      {statePred !== null && lean && (
        <div style={{
          padding: "10px 12px",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          marginBottom: "12px",
          borderLeft: `4px solid ${lean.color}`,
        }}>
          <div style={{ fontFamily: "var(--font-serif)", fontSize: "18px", fontWeight: "700" }}>
            {lean.label}
          </div>
          <div style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            {selectedState} projected Dem share: {(statePred * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            {selectedRace} · Model prior only
          </div>
        </div>
      )}

      {/* ── County Table ────────────────────────────────────────────── */}
      {stateRows.length > 0 && (
        <div>
          <p style={{ margin: "0 0 6px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.4px", color: "var(--color-text-muted)" }}>
            County predictions
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                <th style={{ textAlign: "left", padding: "4px 6px", color: "var(--color-text-muted)", fontWeight: "normal" }}>County</th>
                <th style={{ textAlign: "right", padding: "4px 6px", color: "var(--color-text-muted)", fontWeight: "normal" }}>Dem %</th>
                <th style={{ textAlign: "right", padding: "4px 6px", color: "var(--color-text-muted)", fontWeight: "normal" }}>90% CI</th>
              </tr>
            </thead>
            <tbody>
              {[...stateRows]
                .sort((a, b) => (b.pred_dem_share ?? 0) - (a.pred_dem_share ?? 0))
                .map((row) => {
                  const share = row.pred_dem_share ?? 0;
                  return (
                    <tr key={row.county_fips} style={{ borderBottom: "1px solid var(--color-border)" }}>
                      <td style={{ padding: "5px 6px" }}>{row.county_name ?? row.county_fips}</td>
                      <td style={{
                        padding: "5px 6px", textAlign: "right",
                        color: share > 0.5 ? "var(--color-dem)" : "var(--color-rep)",
                        fontWeight: 600,
                      }}>
                        {(share * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: "5px 6px", textAlign: "right", color: "var(--color-text-muted)" }}>
                        {row.pred_lo90 !== null && row.pred_hi90 !== null
                          ? `${(row.pred_lo90 * 100).toFixed(0)}–${(row.pred_hi90 * 100).toFixed(0)}%`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
