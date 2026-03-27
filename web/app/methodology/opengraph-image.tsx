import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "WetherVane Methodology";

export default function Image() {
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
        {/* Top accent bar — neutral slate */}
        <div
          style={{
            display: "flex",
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 6,
            background: "#444",
          }}
        />

        {/* Main content */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            justifyContent: "center",
          }}
        >
          <div
            style={{
              fontSize: 18,
              color: "#999",
              fontFamily: "Georgia, serif",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              marginBottom: 16,
            }}
          >
            WetherVane
          </div>
          <div
            style={{
              fontSize: 64,
              fontWeight: 700,
              color: "#222",
              fontFamily: "Georgia, serif",
              lineHeight: 1.15,
              marginBottom: 20,
            }}
          >
            Methodology
          </div>
          <div
            style={{
              fontSize: 28,
              color: "#555",
              fontFamily: "Georgia, serif",
              lineHeight: 1.5,
              maxWidth: 760,
              borderLeft: "4px solid #ccc",
              paddingLeft: 20,
            }}
          >
            How we discover electoral communities from shift patterns and
            propagate polling signals across geography
          </div>
        </div>

        {/* Footer branding */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: 16,
            }}
          >
            {["100 fine types", "5 super-types", "3,154 counties", "LOO r = 0.711"].map(
              (label) => (
                <div
                  key={label}
                  style={{
                    display: "flex",
                    padding: "6px 14px",
                    borderRadius: 6,
                    fontSize: 16,
                    background: "white",
                    border: "1px solid #e0e0e0",
                    color: "#444",
                  }}
                >
                  {label}
                </div>
              ),
            )}
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
