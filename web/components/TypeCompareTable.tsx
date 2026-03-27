"use client";
import { useEffect, useState, useCallback } from "react";
import { fetchTypes, fetchTypeDetail, type TypeSummary, type TypeDetail } from "@/lib/api";
import {
  DEMO_DISPLAY,
  DEMO_SKIP,
  formatDemoValue,
  prettifyKey,
  inferFormat,
  formatLean,
  leanColor,
} from "@/lib/typeDisplay";
import { useMapContext } from "@/components/MapContext";

// Ordered list of metric keys for the table rows.
// We show DEMO_DISPLAY keys first (in definition order), then any extras from the API.
function buildRowKeys(details: TypeDetail[]): string[] {
  if (details.length === 0) return [];
  const allKeysSet = new Set<string>();
  for (const d of details) {
    for (const key of Object.keys(d.demographics)) {
      if (!DEMO_SKIP.has(key)) allKeysSet.add(key);
    }
  }
  const allKeys = Array.from(allKeysSet);
  // Put DEMO_DISPLAY keys first (in defined order), then the rest
  const ordered: string[] = [];
  for (const key of Object.keys(DEMO_DISPLAY)) {
    if (allKeysSet.has(key)) ordered.push(key);
  }
  for (const key of allKeys) {
    if (!DEMO_DISPLAY[key]) ordered.push(key);
  }
  return ordered;
}

function computeMinMax(
  keys: string[],
  details: TypeDetail[]
): Map<string, { min: number; max: number }> {
  const result = new Map<string, { min: number; max: number }>();
  for (const key of keys) {
    let min = Infinity;
    let max = -Infinity;
    for (const d of details) {
      const v = d.demographics[key];
      if (v != null) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (min !== Infinity) result.set(key, { min, max });
  }
  return result;
}

/** Return a subtle blue tint intensity 0..1 where 1 = highest value */
function tintAlpha(value: number | undefined, range: { min: number; max: number } | undefined): number {
  if (value == null || range == null) return 0;
  const { min, max } = range;
  if (max === min) return 0;
  return (value - min) / (max - min);
}

interface TypeSelectProps {
  types: TypeSummary[];
  value: number | null;
  onChange: (id: number | null) => void;
  placeholder: string;
}

function TypeSelect({ types, value, onChange, placeholder }: TypeSelectProps) {
  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      aria-label={placeholder}
      style={{
        width: "100%",
        padding: "5px 8px",
        border: "1px solid var(--color-border)",
        borderRadius: "3px",
        fontSize: "12px",
        fontFamily: "var(--font-sans)",
        background: "var(--color-surface)",
        color: "var(--color-text)",
        cursor: "pointer",
      }}
    >
      <option value="">{placeholder}</option>
      {types.map((t) => (
        <option key={t.type_id} value={t.type_id}>
          {t.display_name}
        </option>
      ))}
    </select>
  );
}

export function TypeCompareTable() {
  const { compareTypeIds, setCompareTypeIds } = useMapContext();

  const [allTypes, setAllTypes] = useState<TypeSummary[]>([]);
  const [details, setDetails] = useState<Map<number, TypeDetail>>(new Map());
  const [loadingIds, setLoadingIds] = useState<Set<number>>(new Set());
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortTypeId, setSortTypeId] = useState<number | null>(null);

  // Load all types list once
  useEffect(() => {
    fetchTypes().then((types) => {
      const sorted = [...types].sort((a, b) => b.n_counties - a.n_counties);
      setAllTypes(sorted);

      // Default to the two most populous types if nothing selected yet
      if (compareTypeIds.length === 0 && sorted.length >= 2) {
        setCompareTypeIds([sorted[0].type_id, sorted[1].type_id]);
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch detail for any newly selected type
  useEffect(() => {
    const toFetch = compareTypeIds.filter((id) => !details.has(id) && !loadingIds.has(id));
    if (toFetch.length === 0) return;

    setLoadingIds((prev) => {
      const next = new Set(Array.from(prev));
      toFetch.forEach((id) => next.add(id));
      return next;
    });

    Promise.all(
      toFetch.map((id) =>
        fetchTypeDetail(id)
          .then((d) => ({ id, data: d }))
          .catch(() => ({ id, data: null }))
      )
    ).then((results) => {
      setDetails((prev) => {
        const next = new Map(prev);
        for (const { id, data } of results) {
          if (data) next.set(id, data);
        }
        return next;
      });
      setLoadingIds((prev) => {
        const next = new Set(Array.from(prev));
        results.forEach(({ id }) => next.delete(id));
        return next;
      });
    });
  }, [compareTypeIds]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelectType = useCallback(
    (slot: number, id: number | null) => {
      setCompareTypeIds((prev) => {
        const next = [...prev];
        if (id === null) {
          next.splice(slot, 1);
        } else {
          next[slot] = id;
        }
        // Deduplicate while preserving order
        return next.filter((v, i, a) => a.indexOf(v) === i);
      });
    },
    [setCompareTypeIds]
  );

  const addSlot = useCallback(() => {
    if (compareTypeIds.length < 4) {
      setCompareTypeIds((prev) => [...prev, 0].filter((v, i, a) => a.indexOf(v) === i));
    }
  }, [compareTypeIds, setCompareTypeIds]);

  // Build display details in slot order (skip ids with no detail yet)
  const slotDetails: (TypeDetail | null)[] = compareTypeIds.map((id) => details.get(id) ?? null);

  // The loaded details for table rows
  const loadedDetails = slotDetails.filter((d): d is TypeDetail => d !== null);

  // Build row keys
  const rowKeys = buildRowKeys(loadedDetails);

  // Compute per-metric min/max for tinting
  const minMax = computeMinMax(rowKeys, loadedDetails);

  // Sort rows if a sort column is active
  const sortedRowKeys = sortKey
    ? [...rowKeys].sort((a, b) => {
        if (a === sortKey) return -1;
        if (b === sortKey) return 1;
        return 0;
      })
    : rowKeys;

  // If sortTypeId is set, sort by that type's values descending
  const finalRowKeys =
    sortTypeId !== null
      ? [...rowKeys].sort((a, b) => {
          const typeDetail = details.get(sortTypeId);
          if (!typeDetail) return 0;
          const va = typeDetail.demographics[a] ?? -Infinity;
          const vb = typeDetail.demographics[b] ?? -Infinity;
          return vb - va;
        })
      : sortedRowKeys;

  const colWidth = 420 / (compareTypeIds.length + 1);
  const labelColWidth = Math.max(120, 420 - compareTypeIds.length * 110);
  const typeColWidth = Math.min(120, (420 - labelColWidth) / Math.max(compareTypeIds.length, 1));

  return (
    <div style={{ padding: "12px 16px", fontSize: "12px" }}>
      {/* Type selectors */}
      <div style={{ marginBottom: "10px" }}>
        <p style={{
          margin: "0 0 6px",
          fontSize: "11px",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
          color: "var(--color-text-muted)",
        }}>
          Select types to compare
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          {compareTypeIds.map((id, slot) => (
            <div key={slot} style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              <div style={{ flex: 1 }}>
                <TypeSelect
                  types={allTypes}
                  value={id || null}
                  onChange={(newId) => handleSelectType(slot, newId)}
                  placeholder={`Type ${slot + 1}`}
                />
              </div>
              {compareTypeIds.length > 2 && (
                <button
                  onClick={() => handleSelectType(slot, null)}
                  style={{
                    border: "none",
                    background: "none",
                    cursor: "pointer",
                    color: "var(--color-text-muted)",
                    fontSize: "16px",
                    lineHeight: 1,
                    padding: "0 2px",
                  }}
                  title="Remove"
                >
                  &times;
                </button>
              )}
            </div>
          ))}
          {compareTypeIds.length < 4 && (
            <button
              onClick={addSlot}
              style={{
                alignSelf: "flex-start",
                border: "1px solid var(--color-border)",
                background: "var(--color-surface)",
                borderRadius: "3px",
                padding: "4px 10px",
                fontSize: "12px",
                cursor: "pointer",
                color: "var(--color-text-muted)",
              }}
            >
              + Add type
            </button>
          )}
        </div>
      </div>

      {/* Loading indicator */}
      {loadingIds.size > 0 && (
        <p style={{ color: "var(--color-text-muted)", margin: "8px 0" }}>Loading…</p>
      )}

      {/* Comparison table */}
      {loadedDetails.length >= 1 && (
        <div className="compare-table-scroll" style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed" }}>
            <colgroup>
              <col style={{ width: `${labelColWidth}px` }} />
              {compareTypeIds.map((id) => (
                <col key={id} style={{ width: `${typeColWidth}px` }} />
              ))}
            </colgroup>
            <thead>
              {/* Type name row */}
              <tr style={{ borderBottom: "2px solid var(--color-border)" }}>
                <th style={{ textAlign: "left", padding: "6px 4px", color: "var(--color-text-muted)", fontWeight: "normal", fontSize: "11px" }}>
                  Metric
                </th>
                {compareTypeIds.map((id, slot) => {
                  const d = slotDetails[slot];
                  const isLoading = loadingIds.has(id);
                  return (
                    <th
                      key={id}
                      onClick={() => setSortTypeId(sortTypeId === id ? null : id)}
                      style={{
                        textAlign: "center",
                        padding: "6px 4px",
                        fontWeight: "700",
                        fontSize: "11px",
                        cursor: "pointer",
                        color: sortTypeId === id ? "var(--color-text)" : "var(--color-text-muted)",
                        borderBottom: sortTypeId === id ? "2px solid var(--color-text)" : "2px solid transparent",
                        userSelect: "none",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                      title={d?.display_name ?? undefined}
                    >
                      {isLoading ? "…" : (d?.display_name ?? `Type ${id}`)}
                      {sortTypeId === id && " ▼"}
                    </th>
                  );
                })}
              </tr>
              {/* Lean row */}
              <tr style={{ borderBottom: "1px solid var(--color-border)", background: "var(--color-bg)" }}>
                <td style={{ padding: "4px 4px", color: "var(--color-text-muted)", fontSize: "11px" }}>
                  Partisan lean
                </td>
                {compareTypeIds.map((id, slot) => {
                  const d = slotDetails[slot];
                  return (
                    <td key={id} style={{
                      textAlign: "center",
                      padding: "4px 4px",
                      fontWeight: "700",
                      fontSize: "12px",
                      color: leanColor(d?.mean_pred_dem_share ?? null),
                    }}>
                      {formatLean(d?.mean_pred_dem_share ?? null)}
                    </td>
                  );
                })}
              </tr>
              {/* County count row */}
              <tr style={{ borderBottom: "1px solid var(--color-border)", background: "var(--color-bg)" }}>
                <td style={{ padding: "4px 4px", color: "var(--color-text-muted)", fontSize: "11px" }}>
                  Counties
                </td>
                {compareTypeIds.map((id, slot) => {
                  const d = slotDetails[slot];
                  return (
                    <td key={id} style={{ textAlign: "center", padding: "4px 4px", fontSize: "12px" }}>
                      {d?.n_counties ?? "—"}
                    </td>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {finalRowKeys.map((key, rowIdx) => {
                const display = DEMO_DISPLAY[key];
                const label = display?.label ?? prettifyKey(key);
                const fmt = display?.fmt ?? inferFormat(key);
                const range = minMax.get(key);
                return (
                  <tr
                    key={key}
                    style={{
                      borderBottom: "1px solid var(--color-border)",
                      background: rowIdx % 2 === 0 ? "white" : "#fafafa",
                    }}
                  >
                    <td style={{
                      padding: "4px 4px",
                      color: "var(--color-text-muted)",
                      fontSize: "11px",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}>
                      {label}
                    </td>
                    {compareTypeIds.map((id, slot) => {
                      const d = slotDetails[slot];
                      const value = d?.demographics[key];
                      const alpha = tintAlpha(value, range);
                      return (
                        <td
                          key={id}
                          style={{
                            textAlign: "center",
                            padding: "4px 4px",
                            fontSize: "12px",
                            fontWeight: "500",
                            background:
                              value != null && alpha > 0.05
                                ? `rgba(33, 102, 172, ${alpha * 0.15})`
                                : undefined,
                          }}
                        >
                          {value != null ? formatDemoValue(value, fmt) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {loadedDetails.length === 0 && loadingIds.size === 0 && compareTypeIds.length > 0 && (
        <p style={{ color: "var(--color-text-muted)", fontSize: "13px" }}>
          Select types above to compare.
        </p>
      )}
    </div>
  );
}
