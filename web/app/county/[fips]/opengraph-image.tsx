import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "WetherVane county electoral profile";

const API_BASE = process.env.API_URL || "http://localhost:8002";

interface CountyOG {
  county_name: string | null;
  state_abbr: string;
  type_display_name: string;
  super_type_display_name: string;
  pred_dem_share: number | null;
  demographics: Record<string, number>;
}

function leanLabel(share: number | null): { text: string; color: string } {
  if (share === null) return { text: "N/A", color: "#666666" };
  const margin = Math.abs(share - 0.5) * 100;
  const text = margin < 0.5 ? "EVEN" : share > 0.5 ? `D+${margin.toFixed(0)}` : `R+${margin.toFixed(0)}`;
  const color = share >= 0.5 ? "#2166ac" : "#d73027";
  return { text, color };
}

function fmtPct(v: number | undefined): string {
  return v !== undefined ? `${(v * 100).toFixed(0)}%` : "—";
}

function fmtIncome(v: number | undefined): string {
  return v !== undefined ? `$${Math.round(v).toLocaleString("en-US")}` : "—";
}

function fmtPop(v: number | undefined): string {
  if (v === undefined) return "—";
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
  return Math.round(v).toLocaleString("en-US");
}

function stripCountySuffix(name: string | null): string {
  if (!name) return "Unknown County";
  return name.replace(/,\s*[A-Z]{2}$/, "");
}

const STATE_NAMES: Record<string, string> = {
  AL: "Alabama", AK: "Alaska", AZ: "Arizona", AR: "Arkansas", CA: "California",
  CO: "Colorado", CT: "Connecticut", DE: "Delaware", FL: "Florida", GA: "Georgia",
  HI: "Hawaii", ID: "Idaho", IL: "Illinois", IN: "Indiana", IA: "Iowa",
  KS: "Kansas", KY: "Kentucky", LA: "Louisiana", ME: "Maine", MD: "Maryland",
  MA: "Massachusetts", MI: "Michigan", MN: "Minnesota", MS: "Mississippi",
  MO: "Missouri", MT: "Montana", NE: "Nebraska", NV: "Nevada", NH: "New Hampshire",
  NJ: "New Jersey", NM: "New Mexico", NY: "New York", NC: "North Carolina",
  ND: "North Dakota", OH: "Ohio", OK: "Oklahoma", OR: "Oregon", PA: "Pennsylvania",
  RI: "Rhode Island", SC: "South Carolina", SD: "South Dakota", TN: "Tennessee",
  TX: "Texas", UT: "Utah", VT: "Vermont", VA: "Virginia", WA: "Washington",
  WV: "West Virginia", WI: "Wisconsin", WY: "Wyoming", DC: "District of Columbia",
};

export default async function Image({
  params,
}: {
  params: Promise<{ fips: string }>;
}) {
  const { fips } = await params;

  let data: CountyOG | null = null;
  try {
    const res = await fetch(`${API_BASE}/api/v1/counties/${fips}`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) data = await res.json();
  } catch {
    /* fallback to generic card */
  }

  if (!data) {
    return new ImageResponse(
      (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#f7f8fa",
            fontFamily: "Georgia, serif",
            fontSize: 48,
            color: "#222",
          }}
        >
          WetherVane — County Not Found
        </div>
      ),
      { ...size },
    );
  }

  const name = stripCountySuffix(data.county_name);
  const state = STATE_NAMES[data.state_abbr] || data.state_abbr;
  const lean = leanLabel(data.pred_dem_share);
  const d = data.demographics;

  const stats: { label: string; value: string }[] = [
    { label: "Population", value: fmtPop(d.pop_total) },
    { label: "Median Income", value: fmtIncome(d.median_hh_income) },
    { label: "College+", value: fmtPct(d.pct_bachelors_plus) },
    { label: "White NH", value: fmtPct(d.pct_white_nh) },
  ];

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "#f7f8fa",
          padding: "48px 56px",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Top accent bar */}
        <div
          style={{
            display: "flex",
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            background: lean.color,
          }}
        />

        {/* Header row */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
            <div
              style={{
                fontSize: 52,
                fontWeight: 700,
                color: "#222",
                fontFamily: "Georgia, serif",
                lineHeight: 1.15,
              }}
            >
              {name}
            </div>
            <div
              style={{
                fontSize: 28,
                color: "#666",
                marginTop: 4,
                fontFamily: "Georgia, serif",
              }}
            >
              {state}
            </div>
          </div>

          {/* Political lean badge */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              padding: "16px 32px",
              borderRadius: 12,
              background: "white",
              border: `3px solid ${lean.color}`,
            }}
          >
            <div style={{ fontSize: 48, fontWeight: 800, color: lean.color }}>
              {lean.text}
            </div>
            <div style={{ fontSize: 16, color: "#666", marginTop: 4 }}>
              predicted lean
            </div>
          </div>
        </div>

        {/* Type classification */}
        <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
          <div
            style={{
              display: "flex",
              padding: "8px 18px",
              borderRadius: 6,
              fontSize: 20,
              fontWeight: 600,
              background: "white",
              border: "1px solid #e0e0e0",
              color: "#222",
            }}
          >
            {data.type_display_name}
          </div>
          <div
            style={{
              display: "flex",
              padding: "8px 18px",
              borderRadius: 6,
              fontSize: 20,
              background: "white",
              border: "1px solid #e0e0e0",
              color: "#666",
            }}
          >
            {data.super_type_display_name}
          </div>
        </div>

        {/* Stats grid */}
        <div
          style={{
            display: "flex",
            gap: 24,
            marginTop: "auto",
            marginBottom: 8,
          }}
        >
          {stats.map((s) => (
            <div
              key={s.label}
              style={{
                display: "flex",
                flexDirection: "column",
                flex: 1,
                padding: "16px 20px",
                borderRadius: 8,
                background: "white",
                border: "1px solid #e0e0e0",
              }}
            >
              <div style={{ fontSize: 32, fontWeight: 700, color: "#222" }}>
                {s.value}
              </div>
              <div style={{ fontSize: 16, color: "#666", marginTop: 4 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>

        {/* Footer branding */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 16,
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: "#222",
              fontFamily: "Georgia, serif",
              letterSpacing: "0.02em",
            }}
          >
            WetherVane
          </div>
          <div style={{ fontSize: 16, color: "#999" }}>
            wethervane.hhaines.duckdns.org
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
