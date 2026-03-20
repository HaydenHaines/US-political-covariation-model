import type { Metadata } from "next";
import "./globals.css";
import { MapProvider } from "@/components/MapContext";
import { TabBar } from "@/components/TabBar";

import dynamic from "next/dynamic";
const MapShell = dynamic(() => import("@/components/MapShell"), { ssr: false });

export const metadata: Metadata = {
  title: "Bedrock — 2026 Electoral Forecast",
  description: "Community-based electoral forecasting for the 2026 midterms",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <MapProvider>
          <div style={{
            display: "flex",
            height: "100vh",
            overflow: "hidden",
          }}>
            <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
              <MapShell />
            </div>
            <div style={{
              width: "var(--color-panel-width)",
              display: "flex",
              flexDirection: "column",
              borderLeft: "1px solid var(--color-border)",
              background: "var(--color-surface)",
              overflow: "hidden",
            }}>
              <TabBar />
              <div style={{ flex: 1, overflow: "auto" }}>
                {children}
              </div>
            </div>
          </div>
        </MapProvider>
      </body>
    </html>
  );
}
