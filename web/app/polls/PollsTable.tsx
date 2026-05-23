"use client";

import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

// ── Types ──────────────────────────────────────────────────────────────────

export interface PollEntry {
  race: string;
  geography: string;
  geo_level: string;
  dem_share: number;
  n_sample: number;
  date: string | null;
  pollster: string;
  grade: string | null;
}

interface PollsTableProps {
  polls: PollEntry[];
}

const SORT_KEYS = ["date", "race", "pollster", "dem_share", "n_sample", "grade"] as const;

type SortKey = (typeof SORT_KEYS)[number];
type SortDir = "asc" | "desc";

const DEFAULT_SORT_KEY: SortKey = "date";
const DEFAULT_SORT_DIR: SortDir = "desc";
const SORT_PARAM = "sort";
const DIR_PARAM = "dir";

// ── Helpers ────────────────────────────────────────────────────────────────

/** Grade ordering for sort comparison — lower is better (A > C). */
const GRADE_ORDER: Record<string, number> = {
  "A":  1,
  "A-": 2,
  "B+": 3,
  "B":  4,
  "B-": 5,
  "C+": 6,
  "C":  7,
  "C-": 8,
};

/** Color-code grade: A-tier green-ish, B neutral, B-/C muted, null = muted. */
function gradeColor(grade: string | null): string {
  if (!grade) return "var(--color-text-muted)";
  if (grade === "A" || grade === "A-") return "#4a7c59";
  if (grade === "B+" || grade === "B") return "var(--color-text)";
  return "var(--color-text-muted)";
}

/** Null-safe string comparison — nulls sort last. */
function safeCompare(a: string | null, b: string | null): number {
  if (a === b) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return a.localeCompare(b);
}

/** Format dem_share as a percentage with 1 decimal (e.g. 0.4607 → "46.1%"). */
function formatDemShare(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}

/** Format sample size with comma separators. */
function formatSample(n: number): string {
  return n.toLocaleString();
}

function isSortKey(value: string | null): value is SortKey {
  return SORT_KEYS.includes(value as SortKey);
}

function isSortDir(value: string | null): value is SortDir {
  return value === "asc" || value === "desc";
}

function defaultDirForSort(key: SortKey): SortDir {
  return key === "date" ? "desc" : "asc";
}

function isDefaultSort(key: SortKey, dir: SortDir): boolean {
  return key === DEFAULT_SORT_KEY && dir === DEFAULT_SORT_DIR;
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
          {sortDir === "asc" ? " \u2191" : " \u2193"}
        </span>
      )}
    </th>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function PollsTable({ polls }: PollsTableProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [raceFilter, setRaceFilter] = useState<string>("all");
  const [gradeFilter, setGradeFilter] = useState<string>("all");

  const rawSort = searchParams.get(SORT_PARAM);
  const rawDir = searchParams.get(DIR_PARAM);
  const sortKey = isSortKey(rawSort) ? rawSort : DEFAULT_SORT_KEY;
  const sortDir = isSortKey(rawSort) && isSortDir(rawDir)
    ? rawDir
    : defaultDirForSort(sortKey);

  useEffect(() => {
    if (!rawSort && !rawDir) return;
    if (isSortKey(rawSort) && isSortDir(rawDir) && !isDefaultSort(rawSort, rawDir)) {
      return;
    }

    const params = new URLSearchParams(searchParams.toString());
    if (isDefaultSort(sortKey, sortDir)) {
      params.delete(SORT_PARAM);
      params.delete(DIR_PARAM);
    } else {
      params.set(SORT_PARAM, sortKey);
      params.set(DIR_PARAM, sortDir);
    }

    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }, [pathname, rawDir, rawSort, router, searchParams, sortDir, sortKey]);

  // Derive unique races and grades for the filter dropdowns
  const uniqueRaces = useMemo(
    () => Array.from(new Set(polls.map((p) => p.race))).sort(),
    [polls],
  );
  const uniqueGrades = useMemo(
    () =>
      Array.from(new Set(polls.map((p) => p.grade).filter((g): g is string => g != null))).sort(
        (a, b) => (GRADE_ORDER[a] ?? 99) - (GRADE_ORDER[b] ?? 99),
      ),
    [polls],
  );

  // Filter then sort
  const displayed = useMemo(() => {
    let rows = polls;
    if (raceFilter !== "all") {
      rows = rows.filter((p) => p.race === raceFilter);
    }
    if (gradeFilter !== "all") {
      rows = rows.filter((p) => p.grade === gradeFilter);
    }

    return [...rows].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "date":      cmp = safeCompare(a.date, b.date); break;
        case "race":      cmp = a.race.localeCompare(b.race); break;
        case "pollster":  cmp = a.pollster.localeCompare(b.pollster); break;
        case "dem_share":  cmp = a.dem_share - b.dem_share; break;
        case "n_sample":   cmp = a.n_sample - b.n_sample; break;
        case "grade":
          cmp = (GRADE_ORDER[a.grade ?? ""] ?? 99) - (GRADE_ORDER[b.grade ?? ""] ?? 99);
          break;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [polls, raceFilter, gradeFilter, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    const nextDir =
      sortKey === key ? (sortDir === "asc" ? "desc" : "asc") : defaultDirForSort(key);
    const params = new URLSearchParams(searchParams.toString());

    if (isDefaultSort(key, nextDir)) {
      params.delete(SORT_PARAM);
      params.delete(DIR_PARAM);
    } else {
      params.set(SORT_PARAM, key);
      params.set(DIR_PARAM, nextDir);
    }

    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }

  const isFiltered = raceFilter !== "all" || gradeFilter !== "all";

  return (
    <div>
      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4 items-end">
        <label className="flex flex-col gap-1">
          <span
            className="text-xs font-medium"
            style={{ color: "var(--color-text-muted)" }}
          >
            Race
          </span>
          <select
            value={raceFilter}
            onChange={(e) => setRaceFilter(e.target.value)}
            data-testid="race-filter"
            style={{
              padding: "6px 10px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid var(--color-border)",
              background: "var(--color-surface)",
              color: "var(--color-text)",
            }}
          >
            <option value="all">All Races</option>
            {uniqueRaces.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span
            className="text-xs font-medium"
            style={{ color: "var(--color-text-muted)" }}
          >
            Grade
          </span>
          <select
            value={gradeFilter}
            onChange={(e) => setGradeFilter(e.target.value)}
            data-testid="grade-filter"
            style={{
              padding: "6px 10px",
              fontSize: 13,
              borderRadius: 6,
              border: "1px solid var(--color-border)",
              background: "var(--color-surface)",
              color: "var(--color-text)",
            }}
          >
            <option value="all">All Grades</option>
            {uniqueGrades.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Poll count */}
      <div
        className="mb-3 text-xs"
        data-testid="poll-count"
        style={{ color: "var(--color-text-subtle)" }}
      >
        {displayed.length} poll{displayed.length !== 1 ? "s" : ""}
        {isFiltered ? " (filtered)" : ""} — click any column header to sort
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
          data-testid="polls-table"
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
                label="Date"
                colKey="date"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
                align="left"
              />
              <SortHeader
                label="Race"
                colKey="race"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
                align="left"
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
                label="Dem %"
                colKey="dem_share"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
              <SortHeader
                label="Sample (N)"
                colKey="n_sample"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
              <SortHeader
                label="Grade"
                colKey="grade"
                sortKey={sortKey}
                sortDir={sortDir}
                onSort={handleSort}
              />
            </tr>
          </thead>

          <tbody>
            {displayed.map((row, i) => {
              const rowBg =
                i % 2 === 0 ? "var(--color-bg)" : "var(--color-surface)";
              return (
                <tr
                  key={`${row.date ?? "nodate"}-${row.pollster}-${row.race}-${i}`}
                  style={{
                    borderBottom:
                      i < displayed.length - 1
                        ? "1px solid var(--color-border)"
                        : "none",
                    background: rowBg,
                  }}
                >
                  {/* Date */}
                  <td
                    style={{
                      padding: "10px 14px",
                      color: "var(--color-text-muted)",
                      fontVariantNumeric: "tabular-nums",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {row.date ?? "—"}
                  </td>

                  {/* Race */}
                  <td
                    style={{
                      padding: "10px 14px",
                      color: "var(--color-text)",
                      fontWeight: 500,
                    }}
                  >
                    {row.race}
                  </td>

                  {/* Pollster */}
                  <td
                    style={{
                      padding: "10px 14px",
                      color: "var(--color-text)",
                    }}
                  >
                    {row.pollster}
                  </td>

                  {/* Dem % */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      fontVariantNumeric: "tabular-nums",
                      color: "var(--color-text)",
                    }}
                  >
                    {formatDemShare(row.dem_share)}
                  </td>

                  {/* Sample size */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      fontVariantNumeric: "tabular-nums",
                      color: "var(--color-text-muted)",
                    }}
                  >
                    {formatSample(row.n_sample)}
                  </td>

                  {/* Grade — color-coded */}
                  <td
                    style={{
                      padding: "10px 14px",
                      textAlign: "right",
                      fontWeight: 600,
                      color: gradeColor(row.grade),
                    }}
                  >
                    {row.grade ?? "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
