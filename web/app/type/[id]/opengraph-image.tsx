import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "WetherVane electoral type profile";

const API_BASE = process.env.API_URL || "http://localhost:8002";

interface TypeOG {
  type_id: number;
  display_name: string;
  super_type_display_name?: string;
  mean_pred_dem_share: number | null;
  n_counties: number;
  demographics: Record<string, number>;
  narrative: string | null;
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
  return v !== undefined ? `$${Math.round(v / 1000)}K` : "—";
}

export default async function Image({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let data: TypeOG | null = null;
  try {
    const res = await fetch(`${API_BASE}/api/v1/types/${id}`, {
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
          WetherVane — Type Not Found
        </div>
      ),
      { ...size },
    );
  }

  const lean = leanLabel(data.mean_pred_dem_share);
  const d = data.demographics;

  const stats: { label: string; value: string }[] = [
    { label: "Median Income", value: fmtIncome(d.median_hh_income) },
    { label: "College+", value: fmtPct(d.pct_bachelors_plus) },
    { label: "White NH", value: fmtPct(d.pct_white_nh) },
    { label: "Counties", value: String(data.n_counties) },
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
              {data.display_name}
            </div>
            {data.super_type_display_name && (
              <div
                style={{
                  fontSize: 24,
                  color: "#666",
                  marginTop: 6,
                  fontFamily: "Georgia, serif",
                }}
              >
                {data.super_type_display_name}
              </div>
            )}
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
              mean lean
            </div>
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
