"use client";
import { useEffect, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import { fetchForecast, type ForecastRow } from "@/lib/api";
import { FeedAPoll } from "@/components/FeedAPoll";

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

function ForecastBarChart({ rows }: { rows: ForecastRow[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || rows.length === 0) return;
    const sorted = [...rows]
      .filter((r) => r.pred_dem_share !== null)
      .sort((a, b) => (a.pred_dem_share ?? 0) - (b.pred_dem_share ?? 0));

    const plot = Plot.plot({
      width: ref.current.clientWidth || 380,
      height: 140,
      marginLeft: 6,
      marginRight: 6,
      marginTop: 8,
      marginBottom: 24,
      x: { label: null, axis: null },
      y: { label: "Dem %", domain: [0.2, 0.8], grid: true,
           tickFormat: (d: number) => `${(d * 100).toFixed(0)}%` },
      marks: [
        Plot.ruleY([0.5], { stroke: "#999", strokeDasharray: "4,3" }),
        Plot.rectY(sorted, {
          x: Plot.indexOf,
          y: "pred_dem_share",
          fill: (d: ForecastRow) =>
            (d.pred_dem_share ?? 0) > 0.5 ? "var(--color-dem)" : "var(--color-rep)",
          fillOpacity: 0.7,
        }),
      ],
    });

    ref.current.innerHTML = "";
    ref.current.appendChild(plot);
    return () => { ref.current?.innerHTML && (ref.current.innerHTML = ""); };
  }, [rows]);

  return <div ref={ref} style={{ width: "100%" }} />;
}

export function ForecastView() {
  const [races, setRaces] = useState<string[]>([]);
  const [selectedRace, setSelectedRace] = useState<string>("");
  const [baselineRows, setBaselineRows] = useState<ForecastRow[]>([]);
  const [displayRows, setDisplayRows] = useState<ForecastRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchForecast().then((rows) => {
      const uniqueRaces = Array.from(new Set(rows.map((r) => r.race))).sort();
      setRaces(uniqueRaces);
      if (uniqueRaces.length > 0) setSelectedRace(uniqueRaces[0]);
    });
  }, []);

  useEffect(() => {
    if (!selectedRace) return;
    setLoading(true);
    fetchForecast(selectedRace)
      .then((rows) => {
        setBaselineRows(rows);
        setDisplayRows(rows);
      })
      .finally(() => setLoading(false));
  }, [selectedRace]);

  const selectedState = selectedRace.split(" ")[1] ?? "";
  const stateRows = displayRows.filter((r) => r.state_abbr === selectedState);
  const statePred =
    stateRows.length > 0
      ? stateRows.reduce((sum, r) => sum + (r.pred_dem_share ?? 0), 0) / stateRows.length
      : null;
  const lean = statePred !== null ? getLean(statePred) : null;

  return (
    <div style={{ padding: "16px" }}>
      {/* Race selector */}
      <div style={{ marginBottom: "12px" }}>
        <label style={{ fontSize: "12px", color: "var(--color-text-muted)", display: "block", marginBottom: "4px" }}>
          Race
        </label>
        <select
          value={selectedRace}
          onChange={(e) => setSelectedRace(e.target.value)}
          style={{ padding: "6px 10px", border: "1px solid var(--color-border)", borderRadius: "3px", fontSize: "14px", fontFamily: "var(--font-serif)", background: "white" }}
        >
          {races.map((r) => <option key={r} value={r}>{r.replace("_", " ")}</option>)}
        </select>
      </div>

      {/* State summary */}
      {statePred !== null && lean && (
        <div style={{
          padding: "12px",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          marginBottom: "12px",
          borderLeft: `4px solid ${lean.color}`,
        }}>
          <div style={{ fontFamily: "var(--font-serif)", fontSize: "20px", fontWeight: "700" }}>
            {lean.label}
          </div>
          <div style={{ fontSize: "13px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            Projected {selectedState} dem share: {(statePred * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* Feed-a-Poll */}
      {selectedRace && (
        <FeedAPoll
          state={selectedState}
          race={selectedRace}
          onUpdate={setDisplayRows}
          onReset={() => setDisplayRows(baselineRows)}
        />
      )}

      {/* Bar chart */}
      {loading ? (
        <p style={{ color: "var(--color-text-muted)", fontSize: "13px" }}>Loading…</p>
      ) : (
        <>
          <ForecastBarChart rows={stateRows} />

          {/* County table */}
          <div style={{ marginTop: "12px" }}>
            <p style={{ margin: "0 0 8px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.5px", color: "var(--color-text-muted)" }}>
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
                          fontWeight: "600",
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
        </>
      )}
    </div>
  );
}
