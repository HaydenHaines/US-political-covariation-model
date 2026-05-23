"use client";

import { useEffect } from "react";
import Link from "next/link";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import { stripStateSuffix } from "@/lib/config/states";
import { useUrlState, type UrlStateSpec } from "@/lib/hooks/use-url-state";

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

// ── Constants ──────────────────────────────────────────────────────────────

const SORT_KEYS = ["name", "type", "lean"] as const;
type SortField = (typeof SORT_KEYS)[number];
type SortDir = "asc" | "desc";

interface CountyTableUrlState {
  sort: SortField;
  dir: SortDir;
  page: number;
}

const DEFAULT_SORT_KEY: SortField = "lean";
const DEFAULT_SORT_DIR: SortDir = "desc";
const DEFAULT_PAGE = 0;
const PAGE_SIZE = 25;

const SORT_PARAM = "sort";
const DIR_PARAM = "dir";
const PAGE_PARAM = "page";

// ── Validators ─────────────────────────────────────────────────────────────

function isSortKey(value: string | null): value is SortField {
  return SORT_KEYS.includes(value as SortField);
}

function isSortDir(value: string | null): value is SortDir {
  return value === "asc" || value === "desc";
}

function isValidPage(value: string | null): boolean {
  if (value === null) return false;
  const n = Number(value);
  return Number.isInteger(n) && n >= 0;
}

function defaultDirForSort(key: SortField): SortDir {
  return key === "lean" ? "desc" : "asc";
}

function isDefaultSort(key: SortField, dir: SortDir): boolean {
  return key === DEFAULT_SORT_KEY && dir === DEFAULT_SORT_DIR;
}

const COUNTY_TABLE_URL_SPEC: UrlStateSpec<CountyTableUrlState> = {
  sort: {
    param: SORT_PARAM,
    defaultValue: DEFAULT_SORT_KEY,
    parse: (raw) => (isSortKey(raw) ? raw : DEFAULT_SORT_KEY),
    serialize: (value, state) => (isDefaultSort(value, state.dir) ? null : value),
  },
  dir: {
    param: DIR_PARAM,
    defaultValue: DEFAULT_SORT_DIR,
    parse: (raw, params) => {
      const rawSort = params.get(SORT_PARAM);
      const sort = isSortKey(rawSort) ? rawSort : DEFAULT_SORT_KEY;
      return isSortKey(rawSort) && isSortDir(raw) ? raw : defaultDirForSort(sort);
    },
    serialize: (value, state) => (isDefaultSort(state.sort, value) ? null : value),
  },
  page: {
    param: PAGE_PARAM,
    defaultValue: DEFAULT_PAGE,
    parse: (raw) => (isValidPage(raw) ? Number(raw) : DEFAULT_PAGE),
    serialize: (value) => (value === DEFAULT_PAGE ? null : String(value)),
  },
};

// ── Component ──────────────────────────────────────────────────────────────

export function StateCountyTable({ counties }: StateCountyTableProps) {
  const { state: urlState, update: updateUrlState } = useUrlState(COUNTY_TABLE_URL_SPEC);
  const sortField = urlState.sort;
  const sortDir = urlState.dir;

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
  const page = totalPages > 0 ? Math.min(urlState.page, totalPages - 1) : DEFAULT_PAGE;

  useEffect(() => {
    if (urlState.page !== page) {
      updateUrlState({ page });
    }
  }, [page, updateUrlState, urlState.page]);

  function handleSort(field: SortField) {
    const nextDir =
      sortField === field ? (sortDir === "asc" ? "desc" : "asc") : defaultDirForSort(field);
    updateUrlState({ sort: field, dir: nextDir, page: DEFAULT_PAGE });
  }

  function handlePageChange(newPage: number) {
    const clamped = Math.max(DEFAULT_PAGE, Math.min(newPage, totalPages - 1));
    updateUrlState({ page: clamped });
  }

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
        <table
          data-testid="county-table"
          style={{ width: "100%", borderCollapse: "collapse" }}
        >
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
              onClick={() => handlePageChange(page - 1)}
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
              onClick={() => handlePageChange(page + 1)}
              disabled={page === totalPages - 1}
              style={{
                padding: "4px 12px",
                borderRadius: 4,
                border: "1px solid var(--color-border)",
                background: "var(--color-surface)",
                color:
                  page === totalPages - 1 ? "var(--color-text-muted)" : "var(--color-text)",
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
