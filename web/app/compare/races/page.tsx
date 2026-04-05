import type { Metadata } from "next";
import { Suspense } from "react";
import { RaceComparisonClient } from "./RaceComparisonClient";

export const metadata: Metadata = {
  title: "Race Comparison | WetherVane",
  description:
    "Compare two 2026 election races side-by-side: predictions, polls, electoral type composition, and historical context.",
  openGraph: {
    title: "Race Comparison | WetherVane",
    description:
      "Side-by-side structural model comparison for any two 2026 Senate or Governor races.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "Race Comparison | WetherVane",
    description:
      "Compare WetherVane forecasts for any two 2026 races — polls, type composition, and historical context.",
  },
};

// This page is a client-side shell: URL params drive the race selectors,
// so we cannot do SSR data fetching without knowing which races were chosen.
// The RaceComparisonClient reads `?races=slug1,slug2` from the URL and fetches
// data on the client.  The page component itself is a Server Component (for
// metadata only) with a client child.
export default function RaceComparisonPage() {
  return (
    <Suspense>
      <RaceComparisonClient />
    </Suspense>
  );
}
