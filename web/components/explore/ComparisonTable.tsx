"use client";

/**
 * ComparisonTable — side-by-side comparison of up to 4 electoral types.
 *
 * Rows are driven by the FIELD_DISPLAY config in web/lib/config/display.ts.
 * Color coding for political fields uses partisan colors; demographic fields
 * use a gradient from neutral to saturated based on relative values.
 *
 * URL persistence: ?types=1,5,23,67 keeps selected types across reloads.
 * Mobile: max 2 columns on small screens (CSS grid collapses).
 *
 * CRITICAL: API margins are centered at 0. MarginDisplay expects 0-1 demShare.
 * Add 0.5 when passing margin values from API to MarginDisplay.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
  type ColumnDef,
} from "@tanstack/react-table";
import { X, ChevronsUpDown } from "lucide-react";
import { useTypes } from "@/lib/hooks/use-types";
import { useTypeDetail } from "@/lib/hooks/use-type-detail";
import {
  groupFieldsBySection,
  getFieldConfig,
  SKIP_FIELDS,
} from "@/lib/config/display";
import { formatField, parseMargin } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TypeDetail } from "@/lib/api";
import type { TypeSummary } from "@/lib/types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_TYPES = 4;
const TYPES_PARAM = "types";

// ---------------------------------------------------------------------------
// TypeSelector — searchable combobox for picking one type
// ---------------------------------------------------------------------------

interface TypeSelectorProps {
  types: TypeSummary[];
  selectedIds: number[];
  onSelect: (typeId: number) => void;
  placeholder?: string;
}

function TypeSelector({
  types,
  selectedIds,
  onSelect,
  placeholder = "Add a type to compare…",
}: TypeSelectorProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter out already-selected types and apply search query
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return types.filter((t) => {
      if (selectedIds.includes(t.type_id)) return false;
      if (!q) return true;
      return (
        t.display_name.toLowerCase().includes(q) ||
        String(t.type_id).includes(q)
      );
    });
  }, [types, query, selectedIds]);

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  function handleSelect(typeId: number) {
    onSelect(typeId);
    setQuery("");
    setOpen(false);
    inputRef.current?.blur();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Escape") {
      setOpen(false);
      setQuery("");
    } else if (e.key === "Enter" && filtered.length > 0) {
      handleSelect(filtered[0].type_id);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      const first = listRef.current?.querySelector("li");
      (first as HTMLElement | null)?.focus();
    }
  }

  function handleListKeyDown(
    e: React.KeyboardEvent<HTMLLIElement>,
    typeId: number,
    index: number,
  ) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleSelect(typeId);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      const items = listRef.current?.querySelectorAll("li");
      if (items) (items[index + 1] as HTMLElement | undefined)?.focus();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (index === 0) {
        inputRef.current?.focus();
      } else {
        const items = listRef.current?.querySelectorAll("li");
        if (items) (items[index - 1] as HTMLElement | undefined)?.focus();
      }
    }
  }

  return (
    <div ref={containerRef} style={{ position: "relative", flex: "1 1 200px", minWidth: 0 }}>
      {/* Input trigger */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          border: "1px solid var(--color-border)",
          borderRadius: 6,
          background: "var(--color-surface)",
          paddingLeft: 10,
          paddingRight: 8,
          gap: 4,
        }}
      >
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-autocomplete="list"
          aria-haspopup="listbox"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          style={{
            flex: 1,
            border: "none",
            background: "transparent",
            outline: "none",
            fontSize: 13,
            color: "var(--color-text)",
            padding: "7px 0",
          }}
        />
        <ChevronsUpDown
          size={14}
          style={{ color: "var(--color-text-muted)", flexShrink: 0 }}
          aria-hidden
        />
      </div>

      {/* Dropdown */}
      {open && (
        <ul
          ref={listRef}
          role="listbox"
          aria-label="Electoral types"
          style={{
            position: "absolute",
            top: "calc(100% + 4px)",
            left: 0,
            right: 0,
            zIndex: 50,
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 6,
            boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
            maxHeight: 280,
            overflowY: "auto",
            margin: 0,
            padding: "4px 0",
            listStyle: "none",
          }}
        >
          {filtered.length === 0 && (
            <li
              style={{
                padding: "8px 12px",
                fontSize: 13,
                color: "var(--color-text-muted)",
              }}
            >
              {types.length === 0 ? "Loading…" : "No types match"}
            </li>
          )}
          {filtered.slice(0, 60).map((t, i) => (
            <li
              key={t.type_id}
              role="option"
              tabIndex={-1}
              aria-selected={false}
              onMouseDown={() => handleSelect(t.type_id)}
              onKeyDown={(e) => handleListKeyDown(e, t.type_id, i)}
              style={{
                padding: "6px 12px",
                fontSize: 13,
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                color: "var(--color-text)",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.background =
                  "var(--color-border)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.background = "transparent";
              }}
            >
              <span>{t.display_name}</span>
              <span style={{ fontSize: 11, color: "var(--color-text-muted)", marginLeft: 8 }}>
                #{t.type_id}
              </span>
            </li>
          ))}
          {filtered.length > 60 && (
            <li
              style={{
                padding: "6px 12px",
                fontSize: 12,
                color: "var(--color-text-muted)",
                fontStyle: "italic",
              }}
            >
              {filtered.length - 60} more — type to filter
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TypeDetailLoader — loads detail for a single type id
// ---------------------------------------------------------------------------

function useMultiTypeDetail(ids: number[]): {
  details: (TypeDetail | undefined)[];
  isLoading: boolean;
} {
  const d0 = useTypeDetail(ids[0] ?? null);
  const d1 = useTypeDetail(ids[1] ?? null);
  const d2 = useTypeDetail(ids[2] ?? null);
  const d3 = useTypeDetail(ids[3] ?? null);

  const rawDetails = [d0, d1, d2, d3];
  const details = ids.map((_, i) => rawDetails[i]?.data);
  const isLoading = ids.some((_, i) => rawDetails[i]?.isLoading);

  return { details, isLoading };
}

// ---------------------------------------------------------------------------
// Cell coloring helpers
// ---------------------------------------------------------------------------

/**
 * For a margin (demShare) field, return inline style with partisan color.
 * API margins are centered at 0 — we add 0.5 before calling parseMargin.
 */
function marginCellStyle(value: number | null | undefined): React.CSSProperties {
  if (value == null) return {};
  const { party } = parseMargin(value + 0.5);
  if (party === "dem") return { color: "var(--forecast-safe-d)" };
  if (party === "gop") return { color: "var(--forecast-safe-r)" };
  return {};
}

/**
 * For a demographic field, compute a background color intensity based on
 * how high the value is relative to all selected types (min-max normalized).
 * Returns a CSS background color string (warm amber at high end).
 */
function demographicBgStyle(
  value: number | null | undefined,
  allValues: (number | null | undefined)[],
): React.CSSProperties {
  if (value == null) return {};
  const nums = allValues.filter((v): v is number => v != null);
  if (nums.length < 2) return {};
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  if (max === min) return {};
  const t = (value - min) / (max - min); // 0-1
  // Warm amber gradient: 0 = transparent, 1 = amber tint
  const alpha = t * 0.25;
  return { background: `rgba(195, 155, 25, ${alpha.toFixed(3)})` };
}

// ---------------------------------------------------------------------------
// Table row shape
// ---------------------------------------------------------------------------

interface TableRow {
  fieldKey: string;
  label: string;
  section: string;
  isSectionHeader: boolean;
  values: (number | null | undefined)[];
  fieldFormat: string;
}

// ---------------------------------------------------------------------------
// Main ComparisonTable
// ---------------------------------------------------------------------------

export function ComparisonTable() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Parse selected type IDs from URL
  const initialIds = useMemo((): number[] => {
    const param = searchParams.get(TYPES_PARAM);
    if (!param) return [];
    return param
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n))
      .slice(0, MAX_TYPES);
  }, [searchParams]);

  const [selectedIds, setSelectedIds] = useState<number[]>(initialIds);

  // Sync URL when selectedIds change
  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    if (selectedIds.length > 0) {
      params.set(TYPES_PARAM, selectedIds.join(","));
    } else {
      params.delete(TYPES_PARAM);
    }
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds]);

  const { data: allTypes, isLoading: typesLoading } = useTypes();
  const { details, isLoading: detailsLoading } = useMultiTypeDetail(selectedIds);

  const addType = useCallback((typeId: number) => {
    setSelectedIds((prev) => {
      if (prev.includes(typeId) || prev.length >= MAX_TYPES) return prev;
      return [...prev, typeId];
    });
  }, []);

  const removeType = useCallback((typeId: number) => {
    setSelectedIds((prev) => prev.filter((id) => id !== typeId));
  }, []);

  // Build table rows from the union of all field keys across loaded types
  const tableRows = useMemo((): TableRow[] => {
    const loadedDetails = details.filter((d): d is TypeDetail => d != null);
    if (loadedDetails.length === 0) return [];

    // Collect all demographic field keys present in any loaded type
    const allKeys = new Set<string>();
    for (const d of loadedDetails) {
      for (const k of Object.keys(d.demographics ?? {})) {
        if (!SKIP_FIELDS.has(k)) allKeys.add(k);
      }
    }

    const grouped = groupFieldsBySection(Array.from(allKeys));
    const rows: TableRow[] = [];

    for (const group of grouped) {
      // Section header pseudo-row
      rows.push({
        fieldKey: `__section_${group.section}`,
        label: group.label,
        section: group.section,
        isSectionHeader: true,
        values: [],
        fieldFormat: "raw",
      });

      // Data rows for each field in this section
      for (const { key, config } of group.fields) {
        const values = selectedIds.map((id, idx) => {
          const detail = details[idx];
          if (!detail) return undefined;
          // Political fields: check top-level first, then demographics
          const topLevel = (detail as unknown as Record<string, unknown>)[key];
          if (typeof topLevel === "number") return topLevel;
          return detail.demographics?.[key] ?? null;
        });

        rows.push({
          fieldKey: key,
          label: config.label,
          section: group.section,
          isSectionHeader: false,
          values,
          fieldFormat: config.format,
        });
      }
    }

    return rows;
  }, [details, selectedIds]);

  // TanStack Table setup — use display columns throughout to keep ColumnDef<TableRow>
  // types homogeneous. Accessor columns produce a narrower generic that conflicts
  // when mixed with display columns in a ColumnDef<TableRow>[] array.
  const columnHelper = createColumnHelper<TableRow>();

  const columns = useMemo<ColumnDef<TableRow>[]>(() => {
    const cols: ColumnDef<TableRow>[] = [
      columnHelper.display({
        id: "field",
        header: "Field",
        cell: (info) => {
          const row = info.row.original;
          if (row.isSectionHeader) {
            return (
              <span
                style={{
                  fontFamily: "var(--font-serif)",
                  fontWeight: 700,
                  fontSize: 13,
                  letterSpacing: "0.03em",
                  textTransform: "uppercase",
                  color: "var(--color-text-muted)",
                }}
              >
                {row.label}
              </span>
            );
          }
          return (
            <span style={{ fontSize: 13, color: "var(--color-text)" }}>
              {row.label}
            </span>
          );
        },
      }),
    ];

    // One column per selected type
    selectedIds.forEach((typeId, colIdx) => {
      const detail = details[colIdx];
      const summary = allTypes?.find((t) => t.type_id === typeId);
      const displayName = detail?.display_name ?? summary?.display_name ?? `Type ${typeId}`;

      cols.push(
        columnHelper.display({
          id: `type_${typeId}`,
          header: () => (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: 2,
                minWidth: 0,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  width: "100%",
                  gap: 4,
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-serif)",
                    fontWeight: 700,
                    fontSize: 14,
                    color: "var(--color-text)",
                    lineHeight: 1.2,
                  }}
                >
                  {displayName}
                </span>
                <button
                  onClick={() => removeType(typeId)}
                  aria-label={`Remove ${displayName}`}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    padding: "6px",
                    color: "var(--color-text-muted)",
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    minWidth: 32,
                    minHeight: 32,
                  }}
                >
                  <X size={13} />
                </button>
              </div>
              <span
                style={{
                  fontSize: 11,
                  color: "var(--color-text-muted)",
                  fontFamily: "var(--font-sans)",
                }}
              >
                Type #{typeId}
                {detail && ` · ${detail.n_counties} counties`}
              </span>
            </div>
          ),
          cell: (info) => {
            const row = info.row.original;
            if (row.isSectionHeader) return null;

            const value = row.values[colIdx];

            // Format the display value
            let displayValue: React.ReactNode;
            if (value == null) {
              displayValue = (
                <span style={{ color: "var(--color-text-muted)" }}>—</span>
              );
            } else if (row.fieldFormat === "margin") {
              // API margins are centered at 0; MarginDisplay expects 0-1 demShare
              const demShare = value + 0.5;
              const { text, party } = parseMargin(demShare);
              const color =
                party === "dem"
                  ? "var(--forecast-safe-d)"
                  : party === "gop"
                  ? "var(--forecast-safe-r)"
                  : "var(--forecast-tossup)";
              displayValue = (
                <span
                  style={{
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: 13,
                    fontWeight: 600,
                    color,
                  }}
                >
                  {text}
                </span>
              );
            } else {
              displayValue = (
                <span style={{ fontSize: 13, color: "var(--color-text)" }}>
                  {formatField(row.fieldKey, value)}
                </span>
              );
            }

            // Compute cell background for demographic fields
            const isMarginField = row.fieldFormat === "margin";
            const cellStyle: React.CSSProperties = isMarginField
              ? marginCellStyle(value)
              : demographicBgStyle(value, row.values);

            return (
              <div
                style={{
                  padding: "2px 0",
                  borderRadius: 3,
                  ...cellStyle,
                }}
              >
                {displayValue}
              </div>
            );
          },
        }),
      );
    });

    return cols;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds, details, allTypes]);

  const table = useReactTable({
    data: tableRows,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const canAddMore = selectedIds.length < MAX_TYPES;
  const hasAnyDetail = details.some((d) => d != null);

  return (
    <div>
      {/* Type selector row */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          marginBottom: 20,
          alignItems: "center",
        }}
      >
        {canAddMore && (
          <TypeSelector
            types={allTypes ?? []}
            selectedIds={selectedIds}
            onSelect={addType}
            placeholder={
              selectedIds.length === 0
                ? "Search for a type to compare…"
                : "Add another type…"
            }
          />
        )}
        {selectedIds.length === 0 && !typesLoading && (
          <p
            style={{
              fontSize: 13,
              color: "var(--color-text-muted)",
              margin: 0,
            }}
          >
            Select up to {MAX_TYPES} types to compare side-by-side.
          </p>
        )}
        {selectedIds.length > 0 && (
          <button
            onClick={() => setSelectedIds([])}
            style={{
              fontSize: 12,
              color: "var(--color-text-muted)",
              background: "none",
              border: "none",
              cursor: "pointer",
              textDecoration: "underline",
              padding: "4px 0",
            }}
          >
            Clear all
          </button>
        )}
      </div>

      {/* Loading state */}
      {detailsLoading && selectedIds.length > 0 && !hasAnyDetail && (
        <div
          style={{
            padding: "32px",
            textAlign: "center",
            color: "var(--color-text-muted)",
            fontSize: 13,
          }}
        >
          Loading type details…
        </div>
      )}

      {/* Empty state */}
      {selectedIds.length === 0 && !typesLoading && (
        <div
          style={{
            padding: "40px",
            border: "1px dashed var(--color-border)",
            borderRadius: 8,
            textAlign: "center",
            color: "var(--color-text-muted)",
            fontSize: 14,
            lineHeight: 1.6,
          }}
        >
          <p style={{ margin: "0 0 8px" }}>
            Use the search above to pick 2–4 electoral types.
          </p>
          <p style={{ margin: 0, fontSize: 12 }}>
            Shareable URL is updated automatically as you add types.
          </p>
        </div>
      )}

      {/* Comparison table */}
      {hasAnyDetail && tableRows.length > 0 && (
        <>
          {/* Mobile notice: only showing first 2 types */}
          {selectedIds.length > 2 && (
            <p
              className="md:hidden text-xs mb-2"
              style={{ color: "var(--color-text-muted)" }}
            >
              Showing 2 of {selectedIds.length} types. View on a wider screen to compare all.
            </p>
          )}

          <div
            style={{
              overflowX: "auto",
              border: "1px solid var(--color-border)",
              borderRadius: 8,
            }}
          >
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                tableLayout: "fixed",
                fontSize: 13,
              }}
            >
              <colgroup>
                {/* Field name column */}
                <col style={{ width: "180px" }} />
                {/* Type value columns — equal width, min 140px each.
                    On mobile (<768px), hide columns beyond the first 2 types. */}
                {selectedIds.map((id, colIdx) => (
                  <col
                    key={id}
                    className={colIdx >= 2 ? "max-md:hidden" : undefined}
                    style={{ minWidth: "140px" }}
                  />
                ))}
              </colgroup>

              <thead>
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id}>
                    {hg.headers.map((header, i) => {
                      // Column index 0 = field label, 1+ = type columns (1-indexed → typeColIdx = i-1)
                      const typeColIdx = i - 1;
                      const hiddenOnMobile = i > 0 && typeColIdx >= 2;
                      return (
                        <th
                          key={header.id}
                          className={hiddenOnMobile ? "max-md:hidden" : undefined}
                          style={{
                            padding: i === 0 ? "12px 16px 12px 16px" : "12px 12px",
                            textAlign: i === 0 ? "left" : "left",
                            fontWeight: 600,
                            borderBottom: "2px solid var(--color-border)",
                            background: "var(--color-surface)",
                            verticalAlign: "bottom",
                            position: "sticky",
                            top: 0,
                            zIndex: 1,
                          }}
                        >
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                        </th>
                      );
                    })}
                  </tr>
                ))}
              </thead>

              <tbody>
                {table.getRowModel().rows.map((row, rowIdx) => {
                  const isSectionRow = row.original.isSectionHeader;
                  return (
                    <tr
                      key={row.id}
                      style={{
                        background: isSectionRow
                          ? "var(--color-surface)"
                          : rowIdx % 2 === 0
                          ? "transparent"
                          : "rgba(0,0,0,0.015)",
                        borderTop: isSectionRow
                          ? "1px solid var(--color-border)"
                          : undefined,
                      }}
                    >
                      {row.getVisibleCells().map((cell, cellIdx) => {
                        const isSectionCell = isSectionRow && cellIdx === 0;
                        const isFieldLabel = !isSectionRow && cellIdx === 0;
                        // cellIdx 0 = field label, 1+ = type columns
                        const typeColIdx = cellIdx - 1;
                        const hiddenOnMobile = cellIdx > 0 && typeColIdx >= 2;

                        if (isSectionRow && cellIdx > 0) {
                          // Empty cells on section header row (non-label columns)
                          return (
                            <td
                              key={cell.id}
                              className={hiddenOnMobile ? "max-md:hidden" : undefined}
                              style={{
                                padding: "10px 12px 4px",
                                background: "var(--color-surface)",
                              }}
                            />
                          );
                        }

                        return (
                          <td
                            key={cell.id}
                            className={hiddenOnMobile ? "max-md:hidden" : undefined}
                            style={{
                              padding: isSectionCell
                                ? "10px 16px 4px"
                                : isFieldLabel
                                ? "8px 16px"
                                : "8px 12px",
                              verticalAlign: "middle",
                              color: isFieldLabel
                                ? "var(--color-text-muted)"
                                : "var(--color-text)",
                              borderBottom: "1px solid var(--color-border)",
                            }}
                          >
                            {flexRender(
                              cell.column.columnDef.cell,
                              cell.getContext(),
                            )}
                          </td>
                        );
                      })}
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
