"use client";

import { MapProvider } from "@/components/MapContext";
import { TabBar } from "@/components/TabBar";
import { ThemeToggle } from "@/components/ThemeToggle";
import dynamic from "next/dynamic";

const MapShell = dynamic(() => import("@/components/MapShell"), { ssr: false });

export default function MapLayout({ children }: { children: React.ReactNode }) {
  return (
    <MapProvider>
      <div className="app-shell" style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
      }}>
        <div className="map-pane" style={{ flex: 1, position: "relative", minWidth: 0 }}
          role="region"
          aria-label="Electoral map"
        >
          <MapShell />
          {/* Theme toggle overlaid on map corner */}
          <div style={{
            position: "absolute",
            top: 12,
            right: 12,
            zIndex: 10,
          }}>
            <ThemeToggle />
          </div>
        </div>
        <aside className="panel-pane" style={{
          width: "var(--color-panel-width)",
          display: "flex",
          flexDirection: "column",
          borderLeft: "1px solid var(--color-border)",
          background: "var(--color-surface)",
          overflow: "hidden",
        }}
          role="complementary"
          aria-label="Data panel"
        >
          <TabBar />
          <main id="main-content" className="panel-scroll" style={{ flex: 1, overflow: "auto" }}>
            {children}
          </main>
        </aside>
      </div>
    </MapProvider>
  );
}
