"use client";
import { useMapContext, type LayoutMode } from "@/components/MapContext";
import { DUSTY_INK } from "@/lib/colors";

const MODES: { value: LayoutMode; label: string }[] = [
  { value: "content", label: "Article" },
  { value: "dashboard", label: "Dashboard" },
];

/** Compact segmented control for switching between Article and Dashboard layout. */
export function LayoutToggle() {
  const { layoutMode, setLayoutMode } = useMapContext();

  return (
    <div
      style={{
        display: "inline-flex",
        borderRadius: 6,
        border: `1px solid ${DUSTY_INK.border}`,
        background: DUSTY_INK.cardBg,
        padding: 2,
        gap: 0,
      }}
      role="radiogroup"
      aria-label="Layout mode"
    >
      {MODES.map(({ value, label }) => {
        const active = layoutMode === value;
        return (
          <button
            key={value}
            role="radio"
            aria-checked={active}
            onClick={() => setLayoutMode(value)}
            style={{
              padding: "4px 12px",
              borderRadius: 4,
              border: "none",
              background: active ? DUSTY_INK.text : "transparent",
              color: active ? "#fff" : DUSTY_INK.textMuted,
              fontSize: 12,
              fontWeight: active ? 600 : 400,
              fontFamily: "var(--font-sans)",
              cursor: "pointer",
              transition: "background 150ms, color 150ms",
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
