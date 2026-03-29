import type { Metadata } from "next";
import { MethodologyContent } from "@/components/methodology/MethodologyContent";

// ── Metadata ──────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "Methodology | WetherVane",
  description:
    "How WetherVane discovers electoral communities from shift patterns, estimates type covariance, and propagates polling signals across geography to produce county-level 2026 forecasts.",
  openGraph: {
    title: "Methodology | WetherVane",
    description:
      "How WetherVane discovers electoral communities from shift patterns, estimates type covariance, and propagates polling signals across geography to produce county-level 2026 forecasts.",
    type: "article",
    siteName: "WetherVane",
    images: [
      {
        url: "/methodology/opengraph-image",
        width: 1200,
        height: 630,
        alt: "WetherVane Methodology — How we discover electoral communities",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Methodology | WetherVane",
    description:
      "How WetherVane's type-primary electoral model works: KMeans discovery, soft membership, covariance estimation, and poll propagation.",
  },
};

// ── Page Component ────────────────────────────────────────────────────────

const METHODOLOGY_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "TechArticle",
  headline: "How WetherVane Works",
  description:
    "How WetherVane discovers electoral communities from shift patterns, estimates type covariance, and propagates polling signals across geography to produce county-level 2026 forecasts.",
  url: "https://wethervane.hhaines.duckdns.org/methodology",
  author: {
    "@type": "Person",
    name: "Hayden Haines",
  },
  publisher: {
    "@type": "Organization",
    name: "WetherVane",
    url: "https://wethervane.hhaines.duckdns.org",
  },
  dateModified: "2026-03-27",
  about: [
    { "@type": "Thing", name: "Electoral Forecasting" },
    { "@type": "Thing", name: "KMeans Clustering" },
    { "@type": "Thing", name: "Bayesian Poll Propagation" },
    { "@type": "Thing", name: "US Midterm Elections 2026" },
  ],
  dependencies: "KMeans, Ridge Regression, Histogram Gradient Boosting, Ledoit-Wolf covariance estimation",
  proficiencyLevel: "Expert",
};

export default function MethodologyPage() {
  return (
    <main id="main-content">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(METHODOLOGY_JSON_LD) }}
      />
      <MethodologyContent />
    </main>
  );
}
