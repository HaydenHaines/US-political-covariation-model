import type { Metadata } from "next";
import { NationalProjectionStrip } from "@/components/forecast/NationalProjectionStrip";
import { ELECTION_YEAR } from "@/lib/config/election";

export const metadata: Metadata = {
  title: "Forecast — WetherVane",
  description: `${ELECTION_YEAR} Senate and Governor race forecasts`,
};

export default function ForecastPage() {
  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">
        {ELECTION_YEAR} Election Forecast
      </h1>
      <NationalProjectionStrip />
      <p className="text-sm" style={{ color: "var(--color-text-muted)", lineHeight: 1.6 }}>
        Select a race type above to explore detailed forecasts, polling trends,
        and competitive ratings for each contest.
      </p>
    </div>
  );
}
