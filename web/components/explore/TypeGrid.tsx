"use client";

import { useState, useMemo } from "react";
import { useTypes } from "@/lib/hooks/use-types";
import { useSuperTypes } from "@/lib/hooks/use-super-types";
import { TypeCard } from "./TypeCard";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";

/**
 * Grid of TypeCards grouped by super-type.
 *
 * Includes a search bar that filters by display name (case-insensitive).
 * Super-type names and colors come from the API — nothing is hardcoded here.
 */
export function TypeGrid() {
  const { data: types, isLoading: typesLoading } = useTypes();
  const { data: superTypes, isLoading: stLoading } = useSuperTypes();
  const [query, setQuery] = useState("");

  const isLoading = typesLoading || stLoading;

  // Super-type lookup by ID
  const superTypeMap = useMemo(
    () => new Map((superTypes ?? []).map((st) => [st.super_type_id, st])),
    [superTypes],
  );

  // Filter types by search query
  const filtered = useMemo(() => {
    if (!types) return [];
    const q = query.trim().toLowerCase();
    if (!q) return types;
    return types.filter((t) => t.display_name.toLowerCase().includes(q));
  }, [types, query]);

  // Group filtered types by super_type_id, preserving super-type order
  const groups = useMemo(() => {
    const map = new Map<number, typeof filtered>();
    for (const t of filtered) {
      const existing = map.get(t.super_type_id) ?? [];
      existing.push(t);
      map.set(t.super_type_id, existing);
    }
    // Sort super-type IDs numerically
    return Array.from(map.entries()).sort(([a], [b]) => a - b);
  }, [filtered]);

  if (isLoading) {
    return (
      <div
        className="text-sm"
        style={{ color: "var(--color-text-muted)", padding: "40px 0" }}
      >
        Loading types…
      </div>
    );
  }

  if (!types || types.length === 0) {
    return (
      <div
        className="text-sm"
        style={{
          color: "var(--color-text-muted)",
          padding: "32px",
          border: "1px solid var(--color-border)",
          borderRadius: 8,
          textAlign: "center",
        }}
      >
        Type data unavailable. The model API may be temporarily offline.
      </div>
    );
  }

  return (
    <div>
      {/* Search bar */}
      <div className="mb-8">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search types…"
          className="w-full max-w-sm rounded-md border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-offset-1"
          style={{
            borderColor: "var(--color-border)",
            background: "var(--color-surface)",
            color: "var(--color-text)",
          }}
          aria-label="Search electoral types"
        />
      </div>

      {/* Groups */}
      {groups.length === 0 && (
        <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
          No types match &ldquo;{query}&rdquo;.
        </p>
      )}

      {groups.map(([superTypeId, members]) => {
        const st = superTypeMap.get(superTypeId);
        const stName = st?.display_name ?? `Super-Type ${superTypeId}`;
        const accentHex = rgbToHex(getSuperTypeColor(superTypeId));

        return (
          <section key={superTypeId} className="mb-12">
            {/* Super-type header */}
            <div
              className="flex items-baseline gap-3 mb-4 pb-2"
              style={{ borderBottom: `2px solid ${accentHex}` }}
            >
              <h2
                className="text-xl font-bold m-0"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                <span
                  className="inline-block w-3 h-3 rounded-full mr-2 align-middle"
                  style={{ background: accentHex }}
                  aria-hidden="true"
                />
                {stName}
              </h2>
              <span
                className="text-sm"
                style={{ color: "var(--color-text-muted)" }}
              >
                {members.length} {members.length === 1 ? "type" : "types"} ·{" "}
                {members
                  .reduce((s, t) => s + t.n_counties, 0)
                  .toLocaleString("en-US")}{" "}
                counties
              </span>
            </div>

            {/* Card grid */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
                gap: 12,
              }}
            >
              {members.map((t) => (
                <TypeCard key={t.type_id} type={t} />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
