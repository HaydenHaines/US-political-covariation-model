import { MetadataRoute } from "next";

const BASE_URL = "https://wethervane.hhaines.duckdns.org";
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

// ── Static pages ───────────────────────────────────────────────────────────

const STATIC_PAGES: Array<{
  path: string;
  priority: number;
  changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"];
}> = [
  { path: "",                      priority: 1.0, changeFrequency: "weekly"  },
  { path: "/forecast/senate",      priority: 0.9, changeFrequency: "weekly"  },
  { path: "/forecast/governor",    priority: 0.8, changeFrequency: "weekly"  },
  { path: "/explore/types",        priority: 0.7, changeFrequency: "monthly" },
  { path: "/explore/map",          priority: 0.7, changeFrequency: "weekly"  },
  { path: "/explore/shifts",       priority: 0.6, changeFrequency: "monthly" },
  { path: "/methodology",          priority: 0.6, changeFrequency: "monthly" },
  { path: "/methodology/accuracy", priority: 0.5, changeFrequency: "monthly" },
  { path: "/compare",              priority: 0.6, changeFrequency: "weekly"  },
  { path: "/pollsters",            priority: 0.5, changeFrequency: "monthly" },
  { path: "/changelog",            priority: 0.5, changeFrequency: "weekly"  },
  { path: "/about",                priority: 0.3, changeFrequency: "monthly" },
];

// ── Helpers ────────────────────────────────────────────────────────────────

async function fetchJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { next: { revalidate: 86400 } });
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

// ── Sitemap ────────────────────────────────────────────────────────────────

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const now = new Date();

  // Static pages
  const staticEntries: MetadataRoute.Sitemap = STATIC_PAGES.map(({ path, priority, changeFrequency }) => ({
    url: `${BASE_URL}${path}`,
    lastModified: now,
    changeFrequency,
    priority,
  }));

  // Race pages: /forecast/[slug]
  const raceSlugs = await fetchJson<string[]>("/api/v1/forecast/race-slugs");
  const raceEntries: MetadataRoute.Sitemap = (raceSlugs ?? []).map((slug) => ({
    url: `${BASE_URL}/forecast/${slug}`,
    lastModified: now,
    changeFrequency: "weekly" as const,
    priority: 0.9,
  }));

  // Type pages: /type/[id] — IDs fetched from API so the count reflects the live model
  const types = await fetchJson<Array<{ type_id: number }>>("/api/v1/types");
  const typeEntries: MetadataRoute.Sitemap = (types ?? []).map(({ type_id }) => ({
    url: `${BASE_URL}/type/${type_id}`,
    lastModified: now,
    changeFrequency: "monthly" as const,
    priority: 0.7,
  }));

  // County pages: /county/[fips]
  const counties = await fetchJson<Array<{ county_fips: string }>>("/api/v1/counties");
  const countyEntries: MetadataRoute.Sitemap = (counties ?? []).map(({ county_fips }) => ({
    url: `${BASE_URL}/county/${county_fips}`,
    lastModified: now,
    changeFrequency: "monthly" as const,
    priority: 0.5,
  }));

  // State hub pages: /state/[abbr]
  const STATE_ABBRS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
  ];
  const stateEntries: MetadataRoute.Sitemap = STATE_ABBRS.map((abbr) => ({
    url: `${BASE_URL}/state/${abbr}`,
    lastModified: now,
    changeFrequency: "weekly" as const,
    priority: 0.8,
  }));

  return [...staticEntries, ...raceEntries, ...typeEntries, ...countyEntries, ...stateEntries];
}
