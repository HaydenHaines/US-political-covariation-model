"use client";
import { useEffect, useMemo, useState, useCallback } from "react";
import {
  fetchForecast,
  fetchForecastRaces,
  fetchPolls,
  feedMultiplePolls,
  type ForecastRow,
  type PollRow,
} from "@/lib/api";
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

/** Standard normal CDF via Abramowitz & Stegun approximation (error < 7.5e-8). */
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2315419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p =
    d *
    t *
    (0.3193815 +
      t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))));
  return x > 0 ? 1 - p : p;
}

/**
 * Confidence anchors — maps model confidence (always ≥ 0.5, in the predicted
 * winner) to a familiar analog framed as certainty, not event probability.
 *
 * The uncertainty here is in our model, not in the election outcome. Elections
 * aren't dice rolls — the framing is "how confident are we?" / "how often would
 * a model this confident turn out to be wrong?"
 *
 * Thresholds are midpoints between anchor values. Five buckets cover 50–100%.
 * Two-dice split (58% / 50%) prevents calling every close race "a coin flip."
 */
const CONFIDENCE_ANCHORS: Array<{ min: number; text: string }> = [
  { min: 0.90, text: "about as certain as not rolling snake eyes" },          // ≈ 97%
  { min: 0.79, text: "about as certain as not rolling a 1 on a die" },       // ≈ 83%
  { min: 0.67, text: "about as certain as not drawing a spade" },             //   75%
  { min: 0.54, text: "about as certain as rolling 7 or higher on two dice" },// ≈ 58%
  { min: 0,    text: "about as certain as a coin flip" },                     //   50%
];

/** Confidence is always expressed in the predicted winner (≥ 0.5). */
function getAnchor(winProb: number): string {
  const confidence = Math.max(winProb, 1 - winProb);
  for (const { min, text } of CONFIDENCE_ANCHORS) {
    if (confidence >= min) return text;
  }
  return CONFIDENCE_ANCHORS[CONFIDENCE_ANCHORS.length - 1].text;
}

/**
 * Formats the model error rate for display.
 * e.g. winProb=0.69 → "about 31 of 100 predictions like this"
 */
function formatMisses(winProb: number): string {
  const confidence = Math.max(winProb, 1 - winProb);
  const misses = Math.round((1 - confidence) * 100);
  if (misses <= 0) return "less than 1 of 100 predictions like this";
  return `about ${misses} of 100 predictions like this`;
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

function buildChoropleth(rows: ForecastRow[]): Map<string, number> {
  const m = new Map<string, number>();
  rows.forEach((r) => {
    if (r.pred_dem_share !== null) m.set(r.county_fips, r.pred_dem_share);
  });
  return m;
}

export function ForecastView() {
  const { setForecastState, setForecastChoropleth } = useMapContext();

  const [allRaces, setAllRaces] = useState<string[]>([]);
  // Structural prior from GET /forecast — set on race change, never mutated by recalculate
  const [structuralRows, setStructuralRows] = useState<ForecastRow[]>([]);
  // Currently displayed predictions (structural by default, poll-updated after recalculate)
  const [displayRows, setDisplayRows] = useState<ForecastRow[]>([]);
  const [polls, setPolls] = useState<PollRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasPollUpdate, setHasPollUpdate] = useState(false);

  const [modelPriorOpen, setModelPriorOpen] = useState(true);
  const [statePollsOpen, setStatePollsOpen] = useState(true);

  const [selectedState, setSelectedState] = useState<string>("");
  const YEAR = "2026";
  const [selectedRace, setSelectedRace] = useState<string>("");

  // Load race list once on mount
  useEffect(() => {
    fetchForecastRaces().then(setAllRaces).catch(() => setAllRaces([]));
  }, []);

  // Derive available states from race labels ("2026 FL Senate" → "FL")
  const availableStates = useMemo(() => {
    const stateSet = new Set<string>();
    allRaces.forEach((race) => {
      const s = stateFromRace(race);
      if (s) stateSet.add(s);
    });
    return Array.from(stateSet).sort();
  }, [allRaces]);

  // Auto-select first state when races load
  useEffect(() => {
    if (availableStates.length > 0 && !selectedState) {
      setSelectedState(availableStates[0]);
    }
  }, [availableStates, selectedState]);

  // Races available for selected state
  const racesForState = useMemo(() => {
    if (!selectedState) return [];
    return allRaces.filter((r) => stateFromRace(r) === selectedState).sort();
  }, [allRaces, selectedState]);

  // Auto-select first race when state changes
  useEffect(() => {
    if (racesForState.length > 0) {
      setSelectedRace(racesForState[0]);
    } else {
      setSelectedRace("");
    }
  }, [racesForState]);

  // Sync forecast state to map context for highlight + pan
  useEffect(() => {
    setForecastState(selectedState || null);
  }, [selectedState, setForecastState]);

  // On race change: load structural rows + polls in parallel; reset poll update
  useEffect(() => {
    if (!selectedRace) {
      setStructuralRows([]);
      setDisplayRows([]);
      setPolls([]);
      setHasPollUpdate(false);
      setForecastChoropleth(null);
      return;
    }
    setLoading(true);
    setHasPollUpdate(false);
    Promise.all([
      fetchForecast(selectedRace),
      fetchPolls({ race: selectedRace }),
    ])
      .then(([rows, pollRows]) => {
        setStructuralRows(rows);
        setDisplayRows(rows);
        setPolls(pollRows);
        setForecastChoropleth(buildChoropleth(rows));
      })
      .finally(() => setLoading(false));
  }, [selectedRace]); // eslint-disable-line react-hooks/exhaustive-deps

  // Recalculate: if polls exist, run stacked Bayesian update; else reload structural
  const recalculate = useCallback(async () => {
    if (!selectedRace) return;
    setLoading(true);
    try {
      let rows: ForecastRow[];
      if (polls.length > 0) {
        const result = await feedMultiplePolls({
          cycle: YEAR,
          state: selectedState,
          race: selectedRace,
        });
        rows = result.counties;
        setHasPollUpdate(true);
      } else {
        rows = await fetchForecast(selectedRace);
        setHasPollUpdate(false);
      }
      setDisplayRows(rows);
      setForecastChoropleth(buildChoropleth(rows));
    } finally {
      setLoading(false);
    }
  }, [selectedRace, selectedState, polls, setForecastChoropleth]);

  // Clean up choropleth when leaving
  useEffect(() => () => { setForecastChoropleth(null); setForecastState(null); }, []);

  const stateRows = displayRows.filter((r) => r.state_abbr === selectedState);
  const statePred =
    stateRows.length > 0
      ? stateRows.reduce((sum, r) => sum + (r.pred_dem_share ?? 0), 0) / stateRows.length
      : null;
  const lean = statePred !== null ? getLean(statePred) : null;

  // Win probability: P(D > 0.5) = Φ((μ − 0.5) / σ)
  // Use mean county pred_std as state-level uncertainty; fall back to 0.05 if unavailable.
  const stateStd = stateRows.length > 0
    ? stateRows.reduce((sum, r) => sum + (r.pred_std ?? 0.05), 0) / stateRows.length
    : 0.05;
  const winProb = statePred !== null && stateStd > 0
    ? normCdf((statePred - 0.5) / stateStd)
    : null;

  // Delta lookup: structural prediction per county for delta column
  const structuralMap = useMemo(
    () => new Map(structuralRows.map((r) => [r.county_fips, r.pred_dem_share])),
    [structuralRows]
  );

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

        <SectionHeader
          title={polls.length > 0 ? `State Polls (${polls.length})` : "State Polls"}
          open={statePollsOpen}
          disabled={polls.length === 0}
          onToggle={() => setStatePollsOpen((p) => !p)}
        />
        {statePollsOpen && polls.length > 0 && (
          <div style={{
            border: "1px solid var(--color-border)",
            borderTop: "none",
            borderRadius: "0 0 3px 3px",
            marginBottom: "4px",
          }}>
            {polls.map((poll, i) => (
              <div
                key={i}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto",
                  gap: "6px",
                  padding: "5px 10px",
                  borderBottom: i < polls.length - 1 ? "1px solid var(--color-border)" : undefined,
                  fontSize: "12px",
                  alignItems: "center",
                }}
              >
                <div>
                  <span style={{ color: "#333" }}>{poll.pollster || "Unknown"}</span>
                  {poll.date && (
                    <span style={{ color: "var(--color-text-muted)", fontSize: "11px", marginLeft: "6px" }}>
                      {poll.date.slice(0, 10)}
                    </span>
                  )}
                </div>
                <span style={{ color: "var(--color-text-muted)", fontSize: "11px" }}>
                  n={poll.n_sample.toLocaleString()}
                </span>
                <span style={{
                  fontWeight: 600,
                  color: poll.dem_share > 0.5 ? "var(--color-dem)" : "var(--color-rep)",
                  minWidth: "40px",
                  textAlign: "right",
                }}>
                  {(poll.dem_share * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}
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
        {loading ? "Calculating…" : polls.length > 0 ? `Recalculate (${polls.length} poll${polls.length !== 1 ? "s" : ""})` : "Recalculate"}
      </button>

      {/* ── State Summary ───────────────────────────────────────────── */}
      {statePred !== null && lean && winProb !== null && (
        <div style={{
          padding: "10px 12px",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
          marginBottom: "12px",
          borderLeft: `4px solid ${lean.color}`,
        }}>
          {/* Lean label */}
          <div style={{ fontFamily: "var(--font-serif)", fontSize: "18px", fontWeight: "700" }}>
            {lean.label}
          </div>

          {/* Confidence statement — framed as model certainty, not event probability */}
          <div style={{ fontSize: "12px", marginTop: "5px", lineHeight: "1.5" }}>
            {lean.label === "Toss-up" ? (
              <span style={{ color: "var(--color-text-muted)" }}>
                Too close to call — we&apos;d be wrong about half the time in races this tight
              </span>
            ) : (
              <>
                <span style={{ color: winProb >= 0.5 ? "var(--color-dem)" : "var(--color-rep)", fontWeight: 600 }}>
                  {winProb >= 0.5 ? "Dem" : "Rep"} projected to win
                </span>
                <span style={{ color: "var(--color-text-muted)" }}>
                  {" "}— we&apos;d expect to be wrong in {formatMisses(winProb)}
                </span>
              </>
            )}
          </div>

          {/* Confidence bar (blue = Dem share; marker at 50% = toss-up line) */}
          <div style={{ position: "relative", height: "6px", background: "#f5d0ca", borderRadius: "3px", margin: "7px 0 4px" }}>
            <div style={{
              position: "absolute", left: 0, top: 0, bottom: 0,
              width: `${(winProb * 100).toFixed(1)}%`,
              background: "#2166ac",
              borderRadius: "3px",
              transition: "width 0.4s ease",
            }} />
            {/* toss-up marker */}
            <div style={{
              position: "absolute", left: "50%", top: "-3px", bottom: "-3px",
              width: "1px", background: "#666", opacity: 0.4,
            }} />
          </div>

          {/* Anchoring comparison */}
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", fontStyle: "italic", marginBottom: "6px" }}>
            {getAnchor(winProb)}
          </div>

          {/* Secondary: projected share + source */}
          <div style={{ fontSize: "11px", color: "var(--color-text-muted)", borderTop: "1px solid var(--color-border)", paddingTop: "5px" }}>
            {selectedState} avg Dem share: {(statePred * 100).toFixed(1)}%
            &nbsp;·&nbsp;
            {hasPollUpdate ? `${polls.length} poll${polls.length !== 1 ? "s" : ""} incorporated` : "model prior"}
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
                {hasPollUpdate && (
                  <th style={{ textAlign: "right", padding: "4px 6px", color: "var(--color-text-muted)", fontWeight: "normal" }}>Δ</th>
                )}
                <th style={{ textAlign: "right", padding: "4px 6px", color: "var(--color-text-muted)", fontWeight: "normal" }}>90% CI</th>
              </tr>
            </thead>
            <tbody>
              {[...stateRows]
                .sort((a, b) => (b.pred_dem_share ?? 0) - (a.pred_dem_share ?? 0))
                .map((row) => {
                  const share = row.pred_dem_share ?? 0;
                  const structuralShare = structuralMap.get(row.county_fips) ?? null;
                  const delta = hasPollUpdate && structuralShare !== null ? share - structuralShare : null;
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
                      {hasPollUpdate && (
                        <td style={{
                          padding: "5px 6px", textAlign: "right", fontSize: "11px",
                          color: delta === null ? "var(--color-text-muted)"
                            : delta > 0.002 ? "var(--color-dem)"
                            : delta < -0.002 ? "var(--color-rep)"
                            : "var(--color-text-muted)",
                        }}>
                          {delta === null ? "—"
                            : `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(1)}`}
                        </td>
                      )}
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
