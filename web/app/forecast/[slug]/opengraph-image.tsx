import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "WetherVane race forecast";

const API_BASE = process.env.API_URL || "http://localhost:8002";

interface RaceOG {
  race: string;
  state_abbr: string;
  race_type: string;
  year: number;
  prediction: number | null;
  n_counties: number;
  polls: { dem_share: number }[];
}

const STATE_NAMES: Record<string, string> = {
  AL: "Alabama", AK: "Alaska", AZ: "Arizona", AR: "Arkansas", CA: "California",
  CO: "Colorado", CT: "Connecticut", DE: "Delaware", DC: "District of Columbia",
  FL: "Florida", GA: "Georgia", HI: "Hawaii", ID: "Idaho", IL: "Illinois",
  IN: "Indiana", IA: "Iowa", KS: "Kansas", KY: "Kentucky", LA: "Louisiana",
  ME: "Maine", MD: "Maryland", MA: "Massachusetts", MI: "Michigan",
  MN: "Minnesota", MS: "Mississippi", MO: "Missouri", MT: "Montana",
  NE: "Nebraska", NV: "Nevada", NH: "New Hampshire", NJ: "New Jersey",
  NM: "New Mexico", NY: "New York", NC: "North Carolina", ND: "North Dakota",
  OH: "Ohio", OK: "Oklahoma", OR: "Oregon", PA: "Pennsylvania",
  RI: "Rhode Island", SC: "South Carolina", SD: "South Dakota", TN: "Tennessee",
  TX: "Texas", UT: "Utah", VT: "Vermont", VA: "Virginia", WA: "Washington",
  WV: "West Virginia", WI: "Wisconsin", WY: "Wyoming",
};

function leanLabel(share: number | null): { text: string; color: string } {
  if (share === null) return { text: "No prediction", color: "#666666" };
  const margin = Math.abs(share - 0.5) * 100;
  const text = margin < 0.5 ? "EVEN" : share > 0.5 ? `D+${margin.toFixed(0)}` : `R+${margin.toFixed(0)}`;
  const color = share >= 0.5 ? "#2166ac" : "#d73027";
  return { text, color };
}

export default async function Image({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;

  let data: RaceOG | null = null;
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race/${slug}`, {
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
          WetherVane — Race Not Found
        </div>
      ),
      { ...size },
    );
  }

  const stateName = STATE_NAMES[data.state_abbr] || data.state_abbr;
  const lean = leanLabel(data.prediction);
  const pollCount = data.polls.length;

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
              {data.year} {stateName}
            </div>
            <div
              style={{
                fontSize: 28,
                color: "#666",
                marginTop: 4,
                fontFamily: "Georgia, serif",
              }}
            >
              {data.race_type}
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
              model forecast
            </div>
          </div>
        </div>

        {/* Poll count */}
        <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
          <div
            style={{
              display: "flex",
              padding: "8px 18px",
              borderRadius: 6,
              fontSize: 20,
              background: "white",
              border: "1px solid #e0e0e0",
              color: "#444",
            }}
          >
            {pollCount > 0
              ? `Based on ${pollCount} ${pollCount === 1 ? "poll" : "polls"}`
              : "Structural forecast — no polls yet"}
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
            {data.n_counties} {data.n_counties === 1 ? "county" : "counties"}
          </div>
        </div>

        {/* Spacer */}
        <div style={{ display: "flex", flex: 1 }} />

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
