import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "WetherVane — 2026 Electoral Forecast";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          background: "linear-gradient(135deg, #f7f8fa 0%, #e8ecf1 100%)",
          fontFamily: "Georgia, serif",
        }}
      >
        {/* Top accent: split blue/red */}
        <div
          style={{
            display: "flex",
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 8,
          }}
        >
          <div style={{ flex: 1, background: "#2166ac" }} />
          <div style={{ flex: 1, background: "#d73027" }} />
        </div>

        <div
          style={{
            fontSize: 72,
            fontWeight: 700,
            color: "#222",
            letterSpacing: "-0.02em",
          }}
        >
          WetherVane
        </div>

        <div
          style={{
            fontSize: 28,
            color: "#666",
            marginTop: 12,
            fontFamily: "system-ui, sans-serif",
          }}
        >
          Community-Based Electoral Forecasting
        </div>

        <div
          style={{
            display: "flex",
            gap: 32,
            marginTop: 48,
            fontFamily: "system-ui, sans-serif",
          }}
        >
          {[
            { n: "3,154", label: "Counties" },
            { n: "100", label: "Electoral Types" },
            { n: "50", label: "States + DC" },
          ].map((s) => (
            <div
              key={s.label}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                padding: "16px 32px",
                borderRadius: 8,
                background: "white",
                border: "1px solid #e0e0e0",
              }}
            >
              <div style={{ fontSize: 36, fontWeight: 700, color: "#222" }}>
                {s.n}
              </div>
              <div style={{ fontSize: 16, color: "#666", marginTop: 4 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            position: "absolute",
            bottom: 28,
            fontSize: 18,
            color: "#999",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          wethervane.hhaines.duckdns.org
        </div>
      </div>
    ),
    { ...size },
  );
}
