"use client";
import { ShiftExplorer } from "@/components/ShiftExplorer";
import { TypeCompareTable } from "@/components/TypeCompareTable";
import { useMapContext } from "@/components/MapContext";

export default function ExplorePage() {
  const { compareTypeIds } = useMapContext();
  return (
    <div>
      <ShiftExplorer />
      {compareTypeIds.length > 0 && (
        <div style={{ borderTop: "1px solid var(--color-border, #e0ddd8)", marginTop: 16, paddingTop: 16 }}>
          <TypeCompareTable />
        </div>
      )}
    </div>
  );
}
