/**
 * DemographicsPanel — config-driven demographics display.
 *
 * Iterates Object.entries(demographics), looks up each field in getFieldConfig,
 * groups by section via groupFieldsBySection, and formats values with formatField.
 * New model features auto-display without code changes.
 *
 * Mobile (<768px): single column layout. Desktop (≥768px): two columns.
 */

import { groupFieldsBySection } from "@/lib/config/display";
import { formatField } from "@/lib/format";

interface DemographicsPanelProps {
  demographics: Record<string, number>;
}

export function DemographicsPanel({ demographics }: DemographicsPanelProps) {
  const keys = Object.keys(demographics);
  const sections = groupFieldsBySection(keys);

  if (sections.length === 0) {
    return (
      <p style={{ color: "var(--color-text-muted)", fontSize: 14 }}>
        No demographic data available.
      </p>
    );
  }

  return (
    <div>
      {sections.map(({ section, label, fields }) => (
        <div key={section} style={{ marginBottom: 24 }}>
          <h3
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--color-text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              marginBottom: 8,
            }}
          >
            {label}
          </h3>
          {/* Single column on mobile, two columns on desktop */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-0">
            {fields.map(({ key, config }) => {
              const value = demographics[key];
              return (
                <div
                  key={key}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "5px 0",
                    borderBottom: "1px solid var(--color-bg)",
                    fontSize: 14,
                  }}
                >
                  <span style={{ color: "var(--color-text-muted)" }}>
                    {config.label}
                  </span>
                  <span style={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                    {formatField(key, value)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
