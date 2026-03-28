"use client";
import { createContext, useContext, useState, useCallback, useEffect } from "react";

export type LayoutMode = "content" | "dashboard";

const LAYOUT_MODE_KEY = "wv-layout-mode";

interface MapContextValue {
  selectedCommunityId: number | null;
  setSelectedCommunityId: (id: number | null) => void;
  selectedTypeId: number | null;
  setSelectedTypeId: (id: number | null) => void;
  compareTypeIds: number[];
  setCompareTypeIds: React.Dispatch<React.SetStateAction<number[]>>;
  addToComparison: (id: number) => void;
  // Forecast tab: highlighted state + choropleth data
  forecastState: string | null;
  setForecastState: (s: string | null) => void;
  forecastChoropleth: Map<string, number> | null;
  setForecastChoropleth: (m: Map<string, number> | null) => void;
  // Progressive map: zoomed-into state
  zoomedState: string | null;
  setZoomedState: (s: string | null) => void;
  // Layout mode: content (split pane) or dashboard (full-viewport map)
  layoutMode: LayoutMode;
  setLayoutMode: (m: LayoutMode) => void;
}

const MapContext = createContext<MapContextValue>({
  selectedCommunityId: null,
  setSelectedCommunityId: () => {},
  selectedTypeId: null,
  setSelectedTypeId: () => {},
  compareTypeIds: [],
  setCompareTypeIds: () => {},
  addToComparison: () => {},
  forecastState: null,
  setForecastState: () => {},
  forecastChoropleth: null,
  setForecastChoropleth: () => {},
  zoomedState: null,
  setZoomedState: () => {},
  layoutMode: "content",
  setLayoutMode: () => {},
});

export function MapProvider({ children }: { children: React.ReactNode }) {
  const [selectedCommunityId, setSelectedCommunityId] = useState<number | null>(null);
  const [selectedTypeId, setSelectedTypeId] = useState<number | null>(null);
  const [compareTypeIds, setCompareTypeIds] = useState<number[]>([]);
  const [forecastState, setForecastState] = useState<string | null>(null);
  const [forecastChoropleth, setForecastChoropleth] = useState<Map<string, number> | null>(null);
  const [zoomedState, setZoomedState] = useState<string | null>(null);
  const [layoutMode, setLayoutModeState] = useState<LayoutMode>("content");

  // Hydrate layoutMode from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(LAYOUT_MODE_KEY);
    if (stored === "content" || stored === "dashboard") {
      setLayoutModeState(stored);
    }
  }, []);

  const setLayoutMode = useCallback((mode: LayoutMode) => {
    setLayoutModeState(mode);
    localStorage.setItem(LAYOUT_MODE_KEY, mode);
  }, []);

  const addToComparison = useCallback((id: number) => {
    setCompareTypeIds((prev) => {
      if (prev.includes(id)) return prev;
      if (prev.length >= 4) return [...prev.slice(1), id]; // rotate out oldest if at max
      return [...prev, id];
    });
  }, []);

  return (
    <MapContext.Provider value={{
      selectedCommunityId, setSelectedCommunityId,
      selectedTypeId, setSelectedTypeId,
      compareTypeIds, setCompareTypeIds, addToComparison,
      forecastState, setForecastState,
      forecastChoropleth, setForecastChoropleth,
      zoomedState, setZoomedState,
      layoutMode, setLayoutMode,
    }}>
      {children}
    </MapContext.Provider>
  );
}

export function useMapContext() {
  return useContext(MapContext);
}
