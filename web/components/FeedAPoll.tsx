"use client";
import { useState } from "react";
import { feedPoll, type ForecastRow } from "@/lib/api";
import { formatMargin } from "@/lib/typeDisplay";

interface Props {
  state: string;
  race: string;
  onUpdate: (rows: ForecastRow[]) => void;
  onReset: () => void;
}

export function FeedAPoll({ state, race, onUpdate, onReset }: Props) {
  const [demShare, setDemShare] = useState(0.46);
  const [n, setN] = useState(600);
  const [loading, setLoading] = useState(false);
  const [hasUpdated, setHasUpdated] = useState(false);

  const handleUpdate = async () => {
    setLoading(true);
    try {
      const rows = await feedPoll({ state, race, dem_share: demShare, n });
      onUpdate(rows);
      setHasUpdated(true);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setHasUpdated(false);
    onReset();
  };

  return (
    <div style={{
      border: "1px solid var(--color-border)",
      borderRadius: "4px",
      padding: "14px 16px",
      marginBottom: "16px",
      background: "var(--color-bg)",
    }}>
      <p style={{ margin: "0 0 10px", fontSize: "13px", fontWeight: "600", fontFamily: "var(--font-serif)" }}>
        What if the polls show…
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
        <label style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
          Dem share
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
            <input
              type="range" min={0.25} max={0.75} step={0.01}
              value={demShare}
              onChange={(e) => setDemShare(parseFloat(e.target.value))}
              style={{ width: "120px" }}
            />
            <span style={{ fontWeight: "600", minWidth: "36px" }}>{formatMargin(demShare, 0)}</span>
          </div>
        </label>

        <label style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
          Sample size (n)
          <div style={{ marginTop: "4px" }}>
            <input
              type="number" min={100} max={5000} step={100}
              value={n}
              onChange={(e) => setN(parseInt(e.target.value) || 600)}
              style={{ width: "72px", padding: "3px 6px", border: "1px solid var(--color-border)", borderRadius: "3px" }}
            />
          </div>
        </label>

        <button
          onClick={handleUpdate}
          disabled={loading}
          style={{
            marginTop: "16px",
            padding: "6px 14px",
            background: "var(--color-text)",
            color: "white",
            border: "none",
            borderRadius: "3px",
            cursor: loading ? "wait" : "pointer",
            fontSize: "13px",
          }}
        >
          {loading ? "Updating…" : "Update"}
        </button>

        {hasUpdated && (
          <button
            onClick={handleReset}
            style={{
              marginTop: "16px",
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
      <p style={{ margin: "8px 0 0", fontSize: "11px", color: "var(--color-text-muted)" }}>
        Bayesian update propagated through community covariance structure
      </p>
    </div>
  );
}
