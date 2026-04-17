"use client";

import { useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { MapProvider, useMapContext } from "@/components/MapContext";

// MapShell uses deck.gl which requires browser APIs — must be client-only.
const MapShell = dynamic(() => import("@/components/map/MapShell"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-gray-200 dark:bg-gray-700 animate-pulse" />
  ),
});

/** Tab toggle for switching between Senate and Governor forecasts. */
function ForecastTabs() {
  const pathname = usePathname();
  const isSenate = pathname.startsWith("/forecast/senate") || pathname === "/forecast";
  const isGovernor = pathname.startsWith("/forecast/governor");

  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: "10px 16px",
    fontSize: 14,
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

/**
 * Watches forecastState from MapContext and navigates the right panel
 * to that state's race detail page when a state is clicked on the map.
 */
function ForecastStateNavigator() {
  const { forecastState, setForecastState } = useMapContext();
  const router = useRouter();
  const pathname = usePathname();
  // Track the last state we navigated to, so we don't re-navigate on mount
  const lastNavigated = useRef<string | null>(null);

  useEffect(() => {
    if (!forecastState || forecastState === lastNavigated.current) return;
    lastNavigated.current = forecastState;
    // Build the race slug: 2026-{state_lower}-senate
    const slug = `2026-${forecastState.toLowerCase()}-senate`;
    const target = `/forecast/${slug}`;
    // Don't navigate if we're already on this page
    if (pathname !== target) {
      router.push(target);
    }
  }, [forecastState, router, pathname]);

  return null;
}

export default function ForecastLayout({ children }: { children: React.ReactNode }) {
  return (
    <MapProvider>
      <ForecastStateNavigator />
      <div className="flex flex-col lg:grid lg:grid-cols-[400px_1fr] lg:h-[calc(100vh-3rem)]">
        {/* Panel pane — LEFT on desktop (F-pattern reading), stacks on top on mobile.
            Floating card treatment: ring + shadow + rounded corners + margin give visual
            separation from the map without a hard border cutting the canvas. */}
        <div className="lg:overflow-y-auto lg:z-10 lg:m-3 lg:rounded-xl lg:bg-[var(--color-surface)] lg:ring-1 lg:ring-foreground/10 lg:shadow-lg p-4 lg:p-5">
          <ForecastTabs />
          {children}
        </div>
        {/* Map pane — RIGHT on desktop, hidden on mobile.
            defaultOverlayMode="forecast" keeps the focus on competitive ratings,
            not community-type structure, on all /forecast/* pages. */}
        <div className="hidden lg:block bg-[rgb(var(--color-bg))]">
          <MapShell defaultOverlayMode="forecast" />
        </div>
      </div>
    </MapProvider>
  );
}
