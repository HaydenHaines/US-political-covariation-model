import type { Metadata } from "next";
import Link from "next/link";
import { ELECTION_YEAR } from "@/lib/config/election";

export const metadata: Metadata = {
  title: "Forecast — WetherVane",
  description: `${ELECTION_YEAR} Senate and Governor race forecasts`,
};

export default function ForecastPage() {
  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-2">
        {ELECTION_YEAR} Forecasts
      </h1>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
        Statistical forecasts for the {ELECTION_YEAR} election cycle.
      </p>

      <div className="space-y-3">
        <Link
          href="/forecast/senate"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 4,
            padding: "14px 16px",
            borderRadius: 10,
            border: "1px solid var(--color-border)",
            background: "var(--color-surface)",
            textDecoration: "none",
            transition: "border-color 0.15s, box-shadow 0.15s",
          }}
          className="hover:shadow-sm"
        >
          <span
            className="font-semibold text-base"
            style={{ color: "var(--color-dem)" }}
          >
            Senate
          </span>
          <span className="text-sm" style={{ color: "var(--color-text-muted)" }}>
            {ELECTION_YEAR} U.S. Senate race-by-race forecasts
          </span>
        </Link>

        <Link
          href="/forecast/governor"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 4,
            padding: "14px 16px",
            borderRadius: 10,
            border: "1px solid var(--color-border)",
            background: "var(--color-surface)",
            textDecoration: "none",
            transition: "border-color 0.15s, box-shadow 0.15s",
          }}
          className="hover:shadow-sm"
        >
          <span
            className="font-semibold text-base"
            style={{ color: "var(--color-dem)" }}
          >
            Governor
          </span>
          <span className="text-sm" style={{ color: "var(--color-text-muted)" }}>
            {ELECTION_YEAR} gubernatorial race-by-race forecasts
          </span>
        </Link>
      </div>
    </div>
  );
}
