"use client";
import { createContext, useContext, useState } from "react";

interface MapContextValue {
  selectedCommunityId: number | null;
  setSelectedCommunityId: (id: number | null) => void;
}

const MapContext = createContext<MapContextValue>({
  selectedCommunityId: null,
  setSelectedCommunityId: () => {},
});

export function MapProvider({ children }: { children: React.ReactNode }) {
  const [selectedCommunityId, setSelectedCommunityId] = useState<number | null>(null);
  return (
    <MapContext.Provider value={{ selectedCommunityId, setSelectedCommunityId }}>
      {children}
    </MapContext.Provider>
  );
}

export function useMapContext() {
  return useContext(MapContext);
}
