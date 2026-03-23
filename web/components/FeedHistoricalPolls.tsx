"use client";
import { useState } from "react";
import { feedMultiplePolls, type ForecastRow, type MultiPollResponse } from "@/lib/api";

interface Props {
  state: string;
  race: string;
  onUpdate: (rows: ForecastRow[]) => void;
  onReset: () => void;
}

const CYCLES = ["2020", "2022"];

export function FeedHistoricalPolls({ state, race, onUpdate, onReset }: Props) {
  const [cycle, setCycle] = useState(CYCLES[0]);
  const [raceFilter, setRaceFilter] = useState("");
  const [halfLife, setHalfLife] = useState(30);
  const [applyQuality, setApplyQuality] = useState(true);
  const [loading, setLoading] = useState(false);
  const [hasUpdated, setHasUpdated] = useState(false);
  const [meta, setMeta] = useState<{
    polls_used: number;
    date_range: string;
    effective_n_total: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFeedAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const result: MultiPollResponse = await feedMultiplePolls({
        cycle,
        state,
        race: raceFilter || undefined,
        half_life_days: halfLife,
        apply_quality: applyQuality,
      });
      onUpdate(result.counties);
      setMeta({
        polls_used: result.polls_used,
        date_range: result.date_range,
        effective_n_total: result.effective_n_total,
      });
      setHasUpdated(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch polls");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setHasUpdated(false);
    setMeta(null);
    setError(null);
    onReset();
  };

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: "4px",
        padding: "14px 16px",
        marginBottom: "16px",
        background: "var(--color-bg)",
      }}
    >
      <p
        style={{
          margin: "0 0 10px",
          fontSize: "13px",
          fontWeight: "600",
          fontFamily: "var(--font-serif)",
        }}
      >
        Feed historical polls
      </p>

      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: "12px",
          flexWrap: "wrap",
        }}
      >
        {/* Cycle selector */}
        <label style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
          Cycle
          <div style={{ marginTop: "4px" }}>
            <select
              value={cycle}
              onChange={(e) => setCycle(e.target.value)}
              style={{
                padding: "4px 8px",
                border: "1px solid var(--color-border)",
                borderRadius: "3px",
                fontSize: "13px",
                background: "white",
              }}
            >
              {CYCLES.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </label>

        {/* Race filter */}
        <label style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
          Race filter
          <div style={{ marginTop: "4px" }}>
            <input
              type="text"
              placeholder="e.g. President, Senate"
              value={raceFilter}
              onChange={(e) => setRaceFilter(e.target.value)}
              style={{
                width: "140px",
                padding: "4px 8px",
                border: "1px solid var(--color-border)",
                borderRadius: "3px",
                fontSize: "13px",
              }}
            />
          </div>
        </label>

        {/* Half-life */}
        <label style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
          Half-life (days)
          <div style={{ marginTop: "4px" }}>
            <input
              type="number"
              min={7}
              max={180}
              step={1}
              value={halfLife}
              onChange={(e) => setHalfLife(parseInt(e.target.value) || 30)}
              style={{
                width: "60px",
                padding: "4px 8px",
                border: "1px solid var(--color-border)",
                borderRadius: "3px",
                fontSize: "13px",
              }}
            />
          </div>
        </label>

        {/* Quality weighting toggle */}
        <label
          style={{
            fontSize: "12px",
            color: "var(--color-text-muted)",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            marginBottom: "4px",
          }}
        >
          <input
            type="checkbox"
            checked={applyQuality}
            onChange={(e) => setApplyQuality(e.target.checked)}
          />
          Quality weighting
        </label>

        {/* Feed button */}
        <button
          onClick={handleFeedAll}
          disabled={loading}
          style={{
            padding: "6px 14px",
            background: "var(--color-text)",
            color: "white",
            border: "none",
            borderRadius: "3px",
            cursor: loading ? "wait" : "pointer",
            fontSize: "13px",
          }}
        >
          {loading ? "Loading..." : "Feed All Polls"}
        </button>

        {hasUpdated && (
          <button
            onClick={handleReset}
            style={{
              padding: "6px 14px",
              background: "none",
              color: "var(--color-text-muted)",
              border: "1px solid var(--color-border)",
              borderRadius: "3px",
              cursor: "pointer",
              fontSize: "13px",
            }}
          >
            Reset to baseline
          </button>
        )}
      </div>

      {/* Summary */}
      {meta && (
        <div
          style={{
            marginTop: "10px",
            padding: "8px 12px",
            background: "#f8f9fa",
            borderRadius: "3px",
            fontSize: "12px",
            color: "var(--color-text-muted)",
          }}
        >
          {meta.polls_used} polls, {meta.date_range}, effective N ={" "}
          {meta.effective_n_total.toLocaleString()}
        </div>
      )}

      {error && (
        <div
          style={{
            marginTop: "8px",
            fontSize: "12px",
            color: "#d73027",
          }}
        >
          {error}
        </div>
      )}

      <p
        style={{
          margin: "8px 0 0",
          fontSize: "11px",
          color: "var(--color-text-muted)",
        }}
      >
        Time-decayed, quality-weighted Bayesian update from historical poll data
      </p>
    </div>
  );
}
