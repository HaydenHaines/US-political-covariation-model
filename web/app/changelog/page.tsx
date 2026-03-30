import type { Metadata } from "next";
import { ChangelogContent } from "./ChangelogContent";

export const metadata: Metadata = {
  title: "Forecast Changelog | WetherVane",
  description:
    "Track how WetherVane's 2026 Senate and Governor forecasts change over time as new polls arrive and the model updates.",
  openGraph: {
    title: "Forecast Changelog | WetherVane",
    description:
      "Track how WetherVane's 2026 Senate and Governor forecasts change over time as new polls arrive and the model updates.",
    type: "article",
    siteName: "WetherVane",
  },
};

const CHANGELOG_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "WebPage",
  name: "Forecast Changelog",
  description:
    "Track how WetherVane's 2026 election forecasts change week by week.",
  url: "https://wethervane.hhaines.duckdns.org/changelog",
  isPartOf: {
    "@type": "WebSite",
    name: "WetherVane",
    url: "https://wethervane.hhaines.duckdns.org",
  },
};

export default async function ChangelogPage() {
  const API_URL = process.env.API_URL || "http://localhost:8002";
  let data = null;
  try {
    const res = await fetch(`${API_URL}/api/v1/forecast/changelog`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) data = await res.json();
  } catch {
    // Will render empty state
  }

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(CHANGELOG_JSON_LD) }}
      />
      <ChangelogContent data={data} />
    </>
  );
}
