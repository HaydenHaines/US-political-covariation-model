import { MetadataRoute } from "next";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const base = "https://wethervane.hhaines.duckdns.org";

  // Static pages
  const staticPages = ["", "/forecast", "/about", "/methodology", "/methodology/accuracy", "/compare", "/explore"].map(
    (path) => ({
      url: `${base}${path}`,
      lastModified: new Date(),
      changeFrequency: "weekly" as const,
      priority: path === "" ? 1.0 : 0.8,
    })
  );

  // Fetch all county FIPS from API
  const apiBase =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";
  // Type pages: IDs 0-99 (statically known, no API call needed)
  const typePages = Array.from({ length: 100 }, (_, i) => ({
    url: `${base}/type/${i}`,
    lastModified: new Date(),
    changeFrequency: "monthly" as const,
    priority: 0.7,
  }));

  try {
    const [countiesRes, slugsRes] = await Promise.all([
      fetch(`${apiBase}/api/v1/counties`),
      fetch(`${apiBase}/api/v1/forecast/race-slugs`),
    ]);

    const counties = countiesRes.ok ? await countiesRes.json() : [];
    const raceSlugs: string[] = slugsRes.ok ? await slugsRes.json() : [];

    const countyPages = counties.map((c: { county_fips: string }) => ({
      url: `${base}/county/${c.county_fips}`,
      lastModified: new Date(),
      changeFrequency: "monthly" as const,
      priority: 0.6,
    }));

    const racePages = raceSlugs.map((slug) => ({
      url: `${base}/forecast/${slug}`,
      lastModified: new Date(),
      changeFrequency: "weekly" as const,
      priority: 0.9,
    }));

    return [...staticPages, ...racePages, ...typePages, ...countyPages];
  } catch {
    return [...staticPages, ...typePages];
  }
}
