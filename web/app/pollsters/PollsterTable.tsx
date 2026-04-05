"use client";

import { useState, useMemo } from "react";

// ── Types ──────────────────────────────────────────────────────────────────

export interface PollsterEntry {
  pollster: string;
  rank: number;
  n_polls: number;
  n_races: number;
  rmse_pp: number;
  mean_error_pp: number;
}

interface PollsterTableProps {
  pollsters: PollsterEntry[];
  description: string;
}

type SortKey = "rank" | "pollster" | "n_polls" | "n_races" | "rmse_pp" | "mean_error_pp";
type SortDir = "asc" | "desc";

// ── Helpers ────────────────────────────────────────────────────────────────

/**
 * Color-code bias direction:
 *   Dem-leaning (positive) → blue tint (over-predicted Dem)
 *   GOP-leaning (negative) → red tint (over-predicted Rep)
 *   Near-zero              → neutral text
 */
function biasColor(meanErrorPp: number): string {
  if (Math.abs(meanErrorPp) < 1.0) return "var(--color-text-muted)";
  // Dusty Ink likelyD / likelyR — non-partisan but directional
  return meanErrorPp > 0 ? "#4b6d90" : "#9e5e4e";
}

/**
 * Format bias with a directional label so the sign is intuitive.
 * e.g.  +2.30pp D-lean,  -1.10pp R-lean,  ≈0
 */
function formatBias(meanErrorPp: number): string {
  if (Math.abs(meanErrorPp) < 0.05) return "≈0";
  const abs = Math.abs(meanErrorPp).toFixed(2);
  return meanErrorPp > 0 ? `+${abs}pp D-lean` : `-${abs}pp R-lean`;
}

// ── Sort column header ─────────────────────────────────────────────────────

function SortHeader({
  label,
  colKey,
  sortKey,
  sortDir,
  onSort,
  align = "right",
}: {
  label: string;
  colKey: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (k: SortKey) => void;
  align?: "left" | "right";
}) {
  const active = sortKey === colKey;
  return (
    <th
      onClick={() => onSort(colKey)}
      style={{
        padding: "10px 14px",
        textAlign: align,
        fontWeight: 600,
        color: active ? "var(--color-text)" : "var(--color-text-muted)",
        whiteSpace: "nowrap",
        cursor: "pointer",
        userSelect: "none",
        minWidth: 90,
      }}
    >
      {label}
      {active && (
        <span style={{ fontSize: 11, opacity: 0.7 }}>
          {sortDir === "asc" ? " ↑" : " ↓"}
        </span>
      )}
    </th>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function PollsterTable({ pollsters, description }: PollsterTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  const sorted = useMemo(() => {
    return [...pollsters].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "rank":        cmp = a.rank - b.rank; break;
        case "pollster":    cmp = a.pollster.localeCompare(b.pollster); break;
        case "n_polls":     cmp = a.n_polls - b.n_polls; break;
        case "n_races":     cmp = a.n_races - b.n_races; break;
        case "rmse_pp":     cmp = a.rmse_pp - b.rmse_pp; break;
        case "mean_error_pp": cmp = a.mean_error_pp - b.mean_error_pp; break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [pollsters, sortKey, sortDir]);

  return (
    <div>
      {/* Row count hint */}
      <div
        className="mb-3 text-xs"
        style={{ color: "var(--color-text-subtle)" }}
      >
        {sorted.length} pollster{sorted.length !== 1 ? "s" : ""} — click any
        column header to sort
      </div>

      {/* Responsive table wrapper */}
      <div
        style={{
          overflowX: "auto",
          borderRadius: 8,
          border: "1px solid var(--color-border)",
        }}
      >
        <table
          style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}
        >
          <thead>
            <tr
              style={{
                borderBottom: "2px solid var(--color-border)",
                background: "var(--color-surface)",
              }}
            >
              <SortHeader
                label="Rank"
                colKey="rank"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
                align="right"
              />
              <SortHeader
                label="Pollster"
                colKey="pollster"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
                align="left"
              />
              <SortHeader
                label="Polls"
                colKey="n_polls"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
              <SortHeader
                label="Races"
                colKey="n_races"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
              <SortHeader
                label="RMSE (pp)"
                colKey="rmse_pp"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
              <SortHeader
                label="Mean Bias (pp)"
                colKey="mean_error_pp"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
            </tr>
          </thead>

          <tbody>
            {sorted.map((row, i) => {
              const rowBg =
                i % 2 === 0 ? "var(--color-bg)" : "var(--color-surface)";
              return (
                <tr
                  key={row.pollster}
                  style={{
                    borderBottom:
                      i < sorted.length - 1
                        ? "1px solid var(--color-border)"
                        : "none",
                    background: rowBg,
                  }}
                >
                  {/* Rank */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      color: "var(--color-text-subtle)",
                      fontVariantNumeric: "tabular-nums",
                      width: 56,
                    }}
                  >
                    {row.rank}
                  </td>

                  {/* Pollster name */}
                  <td
                    style={{
                      padding: "10px 14px",
                      color: "var(--color-text)",
                      fontWeight: 500,
                    }}
                  >
                    {row.pollster}
                  </td>

                  {/* Poll count */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      color: "var(--color-text-muted)",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {row.n_polls}
                  </td>

                  {/* Race count */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      color: "var(--color-text-muted)",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {row.n_races}
                  </td>

                  {/* RMSE */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      fontVariantNumeric: "tabular-nums",
                      color: "var(--color-text)",
                    }}
                  >
                    {row.rmse_pp.toFixed(2)}
                  </td>

                  {/* Mean bias — color-coded by direction */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      fontVariantNumeric: "tabular-nums",
                      color: biasColor(row.mean_error_pp),
                      fontWeight: Math.abs(row.mean_error_pp) >= 1.0 ? 600 : 400,
                    }}
                  >
                    {formatBias(row.mean_error_pp)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Data description note */}
      {description && (
        <footer
          className="mt-6 text-xs"
          style={{ color: "var(--color-text-subtle)" }}
        >
          {description}
        </footer>
      )}
    </div>
  );
}
