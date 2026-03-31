"use client";
import Link from "next/link";
import { getColorForSuperType } from "@/components/MapShell";
import { useMapContext } from "@/components/MapContext";
import { formatLean, leanColor } from "@/lib/typeDisplay";
import type { TractFeatureProps } from "@/components/TractPanel";

export interface TractPopupData {
  feature: TractFeatureProps;
  /** Screen x coordinate of the click */
  x: number;
  /** Screen y coordinate of the click */
  y: number;
}

interface Props {
  data: TractPopupData;
  /** Number of counties in this type (from typeDataMap) */
  nCounties: number | null;
  /** Predicted dem share for this type (from typeDataMap) */
  meanDemShare: number | null;
  onClose: () => void;
}

function formatIncome(income: number | null | undefined): string {
  if (income == null) return "\u2014";
  return `$${Math.round(income).toLocaleString("en-US")}`;
}

function formatPct(val: number | null | undefined): string {
  if (val == null) return "\u2014";
  return `${(val * 100).toFixed(1)}%`;
}

function densityLabel(nTracts: number, areaSqkm: number): string {
  // Approximate density from tract count and area
  // Each tract ~4,000 people on average
  const approxPop = nTracts * 4000;
  const density = approxPop / Math.max(areaSqkm * 0.386, 0.1); // convert km2 to mi2
  if (density >= 2000) return "Urban";
  if (density >= 500) return "Suburban";
  if (density >= 100) return "Exurban";
  return "Rural";
}

export function TractPopup({ data, nCounties, meanDemShare, onClose }: Props) {
  const { setSelectedTypeId } = useMapContext();
  const { feature, x, y } = data;
  const superColor = getColorForSuperType(feature.super_type);
  const lean = formatLean(meanDemShare);
  const leanClr = leanColor(meanDemShare);

  // Position the popup so it doesn't overflow the viewport.
  // Prefer placing it to the right and below the click point,
  // but flip if too close to the edge.
  const popupWidth = 280;
  const popupHeight = 300; // approximate
  const offsetX = 16;
  const offsetY = 16;

  const style: React.CSSProperties = {
    position: "absolute",
    zIndex: 50,
    width: popupWidth,
    background: "rgba(20, 24, 32, 0.95)",
    borderRadius: 8,
    padding: "14px 16px 12px",
    fontFamily: "var(--font-sans)",
    boxShadow: "0 8px 24px rgba(0,0,0,0.45), 0 2px 8px rgba(0,0,0,0.2)",
    border: "1px solid rgba(255,255,255,0.08)",
    // Prevent the popup from capturing hover events that would dismiss the tooltip
    pointerEvents: "auto",
  };

  // Determine if we need to flip horizontally or vertically
  // Use window dimensions (safe in client component)
  if (typeof window !== "undefined") {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    style.left = x + offsetX + popupWidth > vw ? x - popupWidth - offsetX : x + offsetX;
    style.top = y + offsetY + popupHeight > vh ? Math.max(8, y - popupHeight - offsetY) : y + offsetY;
  } else {
    style.left = x + offsetX;
    style.top = y + offsetY;
  }

  return (
    <div style={style} onClick={(e) => e.stopPropagation()}>
      {/* Close button */}
      <button
        onClick={onClose}
        aria-label="Close popup"
        style={{
          position: "absolute",
          top: 6,
          right: 8,
          border: "none",
          background: "none",
          cursor: "pointer",
          fontSize: 18,
          color: "rgba(255,255,255,0.5)",
          lineHeight: 1,
          padding: "2px 4px",
        }}
      >
        &times;
      </button>

      {/* Header: super-type color badge + type name */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <div style={{
          width: 12, height: 12, borderRadius: 3,
          background: `rgb(${superColor.join(",")})`,
          flexShrink: 0,
        }} />
        <span style={{
          color: "#f0f4f8",
          fontWeight: 600,
          fontSize: 14,
        }}>
          {feature.super_type_name}
        </span>
      </div>

      <div style={{ color: "rgba(176,190,197,0.8)", fontSize: 11, marginBottom: 10 }}>
        Type {feature.type_id} &middot; {feature.n_tracts.toLocaleString()} tracts
        {nCounties != null && <> &middot; {nCounties} {nCounties === 1 ? "county" : "counties"}</>}
      </div>

      {/* Political lean */}
      <div style={{
        color: leanClr,
        fontWeight: 700,
        fontSize: 16,
        marginBottom: 12,
      }}>
        {lean}
      </div>

      {/* Demographics grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 16px", marginBottom: 12 }}>
        <DemoStat label="Income" value={formatIncome(feature.median_hh_income)} />
        <DemoStat label="College+" value={formatPct(feature.pct_ba_plus)} />
        <DemoStat label="White NH" value={formatPct(feature.pct_white_nh)} />
        <DemoStat label="Density" value={densityLabel(feature.n_tracts, feature.area_sqkm)} />
      </div>

      {/* View type details inline (opens TypePanel in sidebar) */}
      <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 8 }}>
        <button
          onClick={() => {
            setSelectedTypeId(feature.type_id);
            onClose();
          }}
          style={{
            display: "block",
            color: "#6baed6",
            fontSize: 12,
            fontWeight: 500,
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: 0,
            width: "100%",
            textAlign: "left",
          }}
        >
          View type details &rarr;
        </button>
        <Link
          href={`/type/${feature.type_id}`}
          style={{
            display: "inline-block",
            color: "rgba(176,190,197,0.6)",
            fontSize: 11,
            textDecoration: "none",
            marginTop: 4,
          }}
        >
          Full type profile page
        </Link>
      </div>
    </div>
  );
}

function DemoStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ color: "rgba(176,190,197,0.6)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.5px" }}>
        {label}
      </div>
      <div style={{ color: "#e0e6ec", fontSize: 13, fontWeight: 500 }}>
        {value}
      </div>
    </div>
  );
}
