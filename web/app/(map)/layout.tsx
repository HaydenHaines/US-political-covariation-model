"use client";

import { MapProvider, type LayoutMode } from "@/components/MapContext";
import { useMapContext } from "@/components/MapContext";
import { TabBar } from "@/components/TabBar";
import { ThemeToggle } from "@/components/ThemeToggle";
import { LayoutToggle } from "@/components/LayoutToggle";
import dynamic from "next/dynamic";

const MapShell = dynamic(() => import("@/components/MapShell"), { ssr: false });

/** Inner layout that consumes MapContext (must be a child of MapProvider). */
function MapLayoutInner({ children }: { children: React.ReactNode }) {
  const { layoutMode } = useMapContext();
  const isDashboard = layoutMode === "dashboard";

  return (
    <div className="app-shell" style={{
      display: "flex",
      height: "100vh",
      overflow: "hidden",
    }}>
      {/* Map pane — full width in dashboard mode, flex-1 in content mode */}
      <div
        className="map-pane"
        style={{
          flex: isDashboard ? "none" : 1,
          width: isDashboard ? "100%" : undefined,
          position: "relative",
          minWidth: 0,
        }}
        role="region"
        aria-label="Electoral map"
      >
        <MapShell />

        {/* Top-right controls: theme toggle + layout toggle (desktop only) */}
        <div style={{
          position: "absolute",
          top: 12,
          right: 12,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}>
          {/* Hide LayoutToggle on mobile (< 768px) via inline media query class */}
          <div className="layout-toggle-desktop">
            <LayoutToggle />
          </div>
          <ThemeToggle />
        </div>
      </div>

      {/* Right panel — hidden in dashboard mode */}
      {!isDashboard && (
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
      )}

      {/* CSS: hide LayoutToggle on mobile */}
      <style jsx global>{`
        @media (max-width: 767px) {
          .layout-toggle-desktop { display: none !important; }
        }
      `}</style>
    </div>
  );
}

export default function MapLayout({ children }: { children: React.ReactNode }) {
  return (
    <MapProvider>
      <MapLayoutInner>{children}</MapLayoutInner>
    </MapProvider>
  );
}
