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

const SITE_URL = "https://wethervane.hhaines.duckdns.org";

const METHODOLOGY_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "TechArticle",
  headline: "How WetherVane Works",
  description:
    "How WetherVane discovers electoral communities from shift patterns, estimates type covariance, and propagates polling signals across geography to produce county-level 2026 forecasts.",
  url: `${SITE_URL}/methodology`,
  author: {
    "@type": "Person",
    name: "Hayden Haines",
  },
  publisher: {
    "@type": "Organization",
    name: "WetherVane",
    url: SITE_URL,
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

/**
 * FAQPage schema — surfaced as expandable FAQ cards in Google SERPs.
 * Questions are derived from the most common reader questions about the model.
 */
const METHODOLOGY_FAQ_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: [
    {
      "@type": "Question",
      name: "How does WetherVane forecast elections?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "WetherVane discovers electoral communities (called 'types') by clustering counties based on how they shifted politically across multiple elections. It then estimates how those communities covary and propagates current polling signals through that covariance structure to produce county-level forecasts.",
      },
    },
    {
      "@type": "Question",
      name: "What is an electoral type?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "An electoral type is a cluster of counties that move together politically across elections. Types are discovered purely from historical vote-share shifts using KMeans clustering — no demographics are used in the discovery step. Each county has soft (fractional) membership across multiple types.",
      },
    },
    {
      "@type": "Question",
      name: "How is WetherVane different from other forecasters?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "Most forecasters aggregate polls directly. WetherVane instead models the structural covariance between places — if a poll shows movement in one community type, the model infers what that implies for all other counties with similar community composition, even if they haven't been polled.",
      },
    },
    {
      "@type": "Question",
      name: "How accurate is the WetherVane model?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "On leave-one-out cross-validation, the Ridge ensemble achieves a correlation of 0.731 between predicted and actual county-level shifts (LOO r=0.731). This is validated against held-out election pairs to prevent overfitting.",
      },
    },
    {
      "@type": "Question",
      name: "What data sources does WetherVane use?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "WetherVane uses publicly available election returns (2008–2024), ACS demographics, RCMS religious congregation data, CDC health data, BLS industry data, IRS migration flows, Facebook Social Connectedness Index, Census broadband data, and BEA state GDP data — all free, no paid subscriptions.",
      },
    },
    {
      "@type": "Question",
      name: "What races does WetherVane forecast for 2026?",
      acceptedAnswer: {
        "@type": "Answer",
        text: "WetherVane forecasts all competitive 2026 U.S. Senate and Governor races. Forecasts are updated as new polls arrive, typically twice weekly.",
      },
    },
  ],
};

/**
 * Dataset schema describing the underlying election dataset.
 * Helps Google index WetherVane as a data resource alongside journalism/research.
 */
const METHODOLOGY_DATASET_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "Dataset",
  name: "WetherVane Electoral Community Dataset — 2008–2024 County Shifts",
  description:
    "County-level election shift vectors spanning 2008–2024 U.S. presidential, Senate, and gubernatorial elections for 3,154 counties. Used to discover 100 structural electoral community types via KMeans clustering.",
  url: `${SITE_URL}/methodology`,
  creator: {
    "@type": "Organization",
    name: "WetherVane",
    url: SITE_URL,
  },
  temporalCoverage: "2008/2024",
  spatialCoverage: {
    "@type": "Place",
    name: "United States",
    address: { "@type": "PostalAddress", addressCountry: "US" },
  },
  variableMeasured: [
    { "@type": "PropertyValue", name: "Democratic two-party vote share shift", unitText: "percentage points" },
    { "@type": "PropertyValue", name: "Electoral community type", unitText: "cluster ID" },
    { "@type": "PropertyValue", name: "Soft type membership scores", unitText: "probability" },
  ],
  keywords: [
    "US elections",
    "county-level election data",
    "electoral geography",
    "vote shift",
    "2026 midterms",
    "election forecast",
  ],
  license: "https://creativecommons.org/licenses/by/4.0/",
};

export default function MethodologyPage() {
  return (
    <main id="main-content">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(METHODOLOGY_JSON_LD) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(METHODOLOGY_FAQ_JSON_LD) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(METHODOLOGY_DATASET_JSON_LD) }}
      />
      <MethodologyContent />
    </main>
  );
}
