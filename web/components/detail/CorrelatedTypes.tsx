/**
 * CorrelatedTypes — cards for the 3-4 most structurally similar types.
 *
 * "Similar" is defined as sharing the same super_type_id. This is the most
 * principled fallback when no covariance/similarity endpoint exists: super-types
 * are discovered via Ward HAC on type loadings, so co-membership in a super-type
 * means the types moved together historically.
 *
 * Shows a compact card grid with: type name, super-type badge, political lean,
 * and a link to /type/[id].
 *
 * Accepts pre-fetched data as props from the SSR page to avoid client-side
 * loading states. The SWR hooks are intentionally removed: all necessary data
 * is available at render time from the parent server component.
 */

"use client";

import Link from "next/link";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import type { TypeSummary, SuperTypeSummary } from "@/lib/types";

/** Maximum number of sibling types to show. */
const MAX_CORRELATED = 4;

interface CorrelatedTypesProps {
  /** All electoral type summaries — passed from the SSR page. */
  allTypes: TypeSummary[];
  /** All super-type summaries — passed from the SSR page. */
  superTypes: SuperTypeSummary[];
  currentTypeId: number;
  superTypeId: number;
}

interface SiblingCardProps {
  type: TypeSummary;
  superType: SuperTypeSummary | undefined;
}

function SiblingCard({ type, superType }: SiblingCardProps) {
  const accentHex = rgbToHex(getSuperTypeColor(type.super_type_id));
  const stName = superType?.display_name ?? `Super-Type ${type.super_type_id}`;

  return (
    <Link
      href={`/type/${type.type_id}`}
      style={{ textDecoration: "none", display: "block" }}
    >
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 6,
          padding: "12px 14px",
          background: "var(--color-surface, var(--color-bg))",
          transition: "border-color 0.15s",
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLDivElement).style.borderColor = accentHex;
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLDivElement).style.borderColor =
            "var(--color-border)";
        }}
      >
        {/* Super-type badge */}
        <span
          style={{
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: 3,
            fontSize: 11,
            fontWeight: 600,
            background: accentHex + "22",
            border: `1px solid ${accentHex}`,
            color: accentHex,
            marginBottom: 6,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            maxWidth: "100%",
          }}
        >
          {stName}
        </span>

        {/* Type name + lean */}
        <div
          style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}
        >
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                fontSize: 11,
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                marginBottom: 2,
              }}
            >
              Type {type.type_id}
            </div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 700,
                fontFamily: "var(--font-serif)",
                color: "var(--color-text)",
                lineHeight: 1.3,
                wordBreak: "break-word",
              }}
            >
              {type.display_name}
            </div>
          </div>
          <div style={{ flexShrink: 0, paddingTop: 16 }}>
            <MarginDisplay demShare={type.mean_pred_dem_share} size="sm" />
          </div>
        </div>

        {/* County count */}
        <div style={{ fontSize: 12, color: "var(--color-text-muted)", marginTop: 6 }}>
          {type.n_counties} {type.n_counties === 1 ? "county" : "counties"}
        </div>
      </div>
    </Link>
  );
}

export function CorrelatedTypes({
  allTypes,
  superTypes,
  currentTypeId,
  superTypeId,
}: CorrelatedTypesProps) {
  const superTypeMap = new Map(superTypes.map((st) => [st.super_type_id, st]));

  // Siblings: same super-type, excluding current type, up to MAX_CORRELATED
  const siblings = allTypes
    .filter(
      (t) => t.super_type_id === superTypeId && t.type_id !== currentTypeId,
    )
    .slice(0, MAX_CORRELATED);

  if (siblings.length === 0) {
    return (
      <p style={{ fontSize: 14, color: "var(--color-text-muted)" }}>
        No similar types found.
      </p>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
        gap: 12,
      }}
    >
      {siblings.map((t) => (
        <SiblingCard key={t.type_id} type={t} superType={superTypeMap.get(t.super_type_id)} />
      ))}
    </div>
  );
}
