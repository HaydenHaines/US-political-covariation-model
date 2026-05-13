"use client";

import Link from "next/link";
import { useState } from "react";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import { stripStateSuffix } from "@/lib/config/states";

export interface CountyTableRow {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
  dominant_type: number | null;
  super_type: number | null;
  pred_dem_share: number | null;
  type_display_name?: string;
}

interface StateCountyTableProps {
  counties: CountyTableRow[];
}

type SortField = "name" | "type" | "lean";
type SortDir = "asc" | "desc";

export function StateCountyTable({ counties }: StateCountyTableProps) {
  const [sortField, setSortField] = useState<SortField>("lean");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 25;

  function handleSort(field: SortField) {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir(field === "lean" ? "desc" : "asc");
    }
    setPage(0);
  }

  const sorted = [...counties].sort((a, b) => {
    let cmp = 0;
    if (sortField === "name") {
      const an = stripStateSuffix(a.county_name);
      const bn = stripStateSuffix(b.county_name);
      cmp = an.localeCompare(bn);
    } else if (sortField === "type") {
      const at = a.type_display_name ?? "";
      const bt = b.type_display_name ?? "";
      cmp = at.localeCompare(bt);
    } else {
      const av = a.pred_dem_share ?? 0.5;
      const bv = b.pred_dem_share ?? 0.5;
      cmp = av - bv;
    }
    return sortDir === "asc" ? cmp : -cmp;
  });

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
  const slice = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const thStyle: React.CSSProperties = {
    padding: "8px 12px",
    textAlign: "left",
    fontSize: 12,
    fontWeight: 600,
    color: "var(--color-text-muted)",
    borderBottom: "1px solid var(--color-border)",
    cursor: "pointer",
    userSelect: "none",
    whiteSpace: "nowrap",
  };

  const sortArrow = (field: SortField) => {
    if (sortField !== field) return " ↕";
    return sortDir === "asc" ? " ↑" : " ↓";
  };

  return (
    <div>
      <div
        style={{
          overflowX: "auto",
          borderRadius: 8,
          border: "1px solid var(--color-border)",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "var(--color-surface)" }}>
              <th style={thStyle} onClick={() => handleSort("name")}>
                County{sortArrow("name")}
              </th>
              <th style={thStyle} onClick={() => handleSort("type")}>
                Type{sortArrow("type")}
              </th>
              <th
                style={{ ...thStyle, textAlign: "right" }}
                onClick={() => handleSort("lean")}
              >
                Predicted Lean{sortArrow("lean")}
              </th>
            </tr>
          </thead>
          <tbody>
            {slice.map((c, i) => {
              const superType = c.super_type ?? (c.dominant_type ? c.dominant_type % 8 : 0);
              const typeColor = rgbToHex(getSuperTypeColor(superType));
              const name = stripStateSuffix(c.county_name);

              return (
                <tr
                  key={c.county_fips}
                  style={{
                    background: i % 2 === 0 ? "var(--color-bg)" : "var(--color-surface)",
                    borderBottom: "1px solid var(--color-border)",
                  }}
                >
                  <td style={{ padding: "8px 12px" }}>
                    <Link
                      href={`/county/${c.county_fips}`}
                      style={{
                        color: "var(--color-dem)",
                        textDecoration: "none",
                        fontSize: 14,
                        fontWeight: 500,
                      }}
                    >
                      {name}
                    </Link>
                  </td>
                  <td style={{ padding: "8px 12px" }}>
                    {c.dominant_type != null ? (
                      <Link
                        href={`/type/${c.dominant_type}`}
                        style={{
                          display: "inline-block",
                          padding: "2px 8px",
                          borderRadius: 4,
                          fontSize: 12,
                          fontWeight: 500,
                          background: typeColor + "22",
                          border: `1px solid ${typeColor}`,
                          color: typeColor,
                          textDecoration: "none",
                          whiteSpace: "nowrap",
                          maxWidth: 200,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {c.type_display_name ?? `Type ${c.dominant_type}`}
                      </Link>
                    ) : (
                      <span style={{ color: "var(--color-text-muted)", fontSize: 13 }}>—</span>
                    )}
                  </td>
                  <td style={{ padding: "8px 12px", textAlign: "right" }}>
                    <MarginDisplay demShare={c.pred_dem_share} size="sm" />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 12,
            fontSize: 13,
            color: "var(--color-text-muted)",
          }}
        >
          <span>
            {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)} of{" "}
            {sorted.length} counties
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              style={{
                padding: "4px 12px",
                borderRadius: 4,
                border: "1px solid var(--color-border)",
                background: "var(--color-surface)",
                color: page === 0 ? "var(--color-text-muted)" : "var(--color-text)",
                cursor: page === 0 ? "not-allowed" : "pointer",
                fontSize: 13,
              }}
            >
              Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page === totalPages - 1}
              style={{
                padding: "4px 12px",
                borderRadius: 4,
                border: "1px solid var(--color-border)",
                background: "var(--color-surface)",
                color: page === totalPages - 1 ? "var(--color-text-muted)" : "var(--color-text)",
                cursor: page === totalPages - 1 ? "not-allowed" : "pointer",
                fontSize: 13,
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
