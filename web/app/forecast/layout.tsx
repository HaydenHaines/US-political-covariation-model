"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { MapProvider } from "@/components/MapContext";

// MapShell uses deck.gl which requires browser APIs — must be client-only.
const MapShell = dynamic(() => import("@/components/map/MapShell"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-[rgb(var(--color-text-muted))]">
      Loading map…
    </div>
  ),
});

/** Tab toggle for switching between Senate and Governor forecasts. */
function ForecastTabs() {
  const pathname = usePathname();
  const isSenate = pathname.startsWith("/forecast/senate") || pathname === "/forecast";
  const isGovernor = pathname.startsWith("/forecast/governor");

  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: "6px 16px",
    fontSize: 13,
    fontWeight: 600,
    borderRadius: 6,
    textDecoration: "none",
    transition: "background 0.15s, color 0.15s",
    background: active ? "var(--color-dem)" : "transparent",
    color: active ? "#fff" : "var(--color-text-muted)",
    border: active ? "1px solid var(--color-dem)" : "1px solid var(--color-border)",
  });

  return (
    <div
      style={{
        display: "flex",
        gap: 6,
        marginBottom: 20,
        paddingBottom: 16,
        borderBottom: "1px solid var(--color-border)",
      }}
    >
      <Link href="/forecast/senate" style={tabStyle(isSenate)}>
        Senate
      </Link>
      <Link href="/forecast/governor" style={tabStyle(isGovernor)}>
        Governor
      </Link>
    </div>
  );
}

export default function ForecastLayout({ children }: { children: React.ReactNode }) {
  return (
    <MapProvider>
      <div className="flex flex-col lg:flex-row h-[calc(100vh-3rem)]">
        {/* Map pane — desktop only; hidden on mobile */}
        <div className="hidden lg:block lg:w-1/2 bg-[rgb(var(--color-bg))] border-r border-[rgb(var(--color-border))]">
          <MapShell />
        </div>
        {/* Panel pane */}
        <div className="flex-1 overflow-y-auto p-4 lg:p-6">
          <ForecastTabs />
          {children}
        </div>
      </div>
    </MapProvider>
  );
}
