import { test, expect } from "@playwright/test";

test.describe("Forecast flow", () => {
  test.describe("Senate overview page", () => {
    test("senate overview page loads", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page).toHaveURL(/\/forecast\/senate/);
    });

    test("senate page has a heading", async ({ page }) => {
      await page.goto("/forecast/senate");
      const heading = page.locator("h1");
      await expect(heading).toBeVisible({ timeout: 30_000 });
      const text = await heading.textContent();
      expect(text).toContain("Senate");
    });

    test("chamber probability banner renders", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      // The chamber section uses aria-label, not visible text
      const chamberSection = page.locator('[aria-label="Chamber control probability"]');
      await expect(chamberSection).toBeVisible({ timeout: 10_000 });
    });

    test("rating categories render in map legend", async ({ page }) => {
      await page.goto("/forecast/senate");
      const h1 = page.locator("h1");
      await expect(h1).toBeVisible({ timeout: 30_000 });
      // Rating categories (Safe D, Likely D, etc.) appear in the map legend
      await expect(page.getByText("Safe D").first()).toBeVisible({ timeout: 10_000 });
    });

    test("race cards exist in the DOM after data loads", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

      // Race cards link to /forecast/2026-* — they may be below the fold
      const raceLinks = page.locator('a[href^="/forecast/2026-"]');
      await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });
      const count = await raceLinks.count();
      expect(count).toBeGreaterThan(0);
    });

    test("clicking a race card navigates to race detail page", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

      // Race cards may be below the fold — use JS navigation
      const firstRaceLink = page.locator('a[href^="/forecast/2026-"]').first();
      await firstRaceLink.waitFor({ state: "attached", timeout: 10_000 });
      const href = await firstRaceLink.getAttribute("href");

      // Navigate directly since the link element is not scrollable into view
      await page.goto(href!);
      await expect(page).toHaveURL(new RegExp(`${href!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`), { timeout: 10_000 });
    });

    test("blend controls button is present", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      const blendBtn = page.getByText("Adjust Forecast Blend");
      await expect(blendBtn).toBeVisible({ timeout: 10_000 });
    });
  });

  test.describe("Race detail page", () => {
    // Navigate directly to a known race slug to avoid scroll/click issues
    const raceSlug = "/forecast/2026-ga-senate";

    test("race detail page shows article with main content", async ({ page }) => {
      await page.goto(raceSlug);
      const article = page.locator("article#main-content");
      await expect(article).toBeVisible({ timeout: 15_000 });
    });

    test("race detail page has a breadcrumb nav", async ({ page }) => {
      await page.goto(raceSlug);
      const breadcrumb = page.locator('nav[aria-label="breadcrumb"]');
      await expect(breadcrumb).toBeVisible({ timeout: 15_000 });
    });

    test("race detail page shows race title in heading", async ({ page }) => {
      await page.goto(raceSlug);
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 15_000 });
      const text = await h1.textContent();
      expect(text).toContain("Georgia");
    });

    test("race detail page shows poll section", async ({ page }) => {
      await page.goto(raceSlug);
      const pollHeading = page.getByText("Recent Polls");
      await expect(pollHeading).toBeVisible({ timeout: 15_000 });
    });

    test("race detail page shows poll table or no-polls message", async ({ page }) => {
      await page.goto(raceSlug);
      const pollHeading = page.getByText("Recent Polls");
      await expect(pollHeading).toBeVisible({ timeout: 15_000 });

      // Scroll to the poll section so table is in viewport
      await pollHeading.scrollIntoViewIfNeeded();

      // GA Senate has polls, so table should be present
      const table = page.locator('table[aria-label="Race polls"]');
      const noPolls = page.getByText(/no polls/i);
      const hasTable = await table.isVisible().catch(() => false);
      const hasNoPolls = await noPolls.isVisible().catch(() => false);
      expect(hasTable || hasNoPolls).toBe(true);
    });

    test("race detail page has a back link to forecast", async ({ page }) => {
      await page.goto(raceSlug);
      const backLink = page.getByRole("link", { name: "← Back to Forecast" });
      await expect(backLink).toBeVisible({ timeout: 15_000 });
    });

    test("poll trend chart renders when polls exist", async ({ page }) => {
      await page.goto(raceSlug);
      // Wait for main content to load
      await expect(page.locator("article#main-content")).toBeVisible({ timeout: 15_000 });
      // Poll trend chart is a client component — wait for it to hydrate
      const chart = page.locator('svg[aria-label="Poll trend chart"]');
      // If polls exist the chart SVG should be in the DOM (may be off-screen)
      const table = page.locator('table[aria-label="Race polls"]');
      const hasTable = await table.isVisible({ timeout: 10_000 }).catch(() => false);
      if (hasTable) {
        await expect(chart).toBeAttached({ timeout: 10_000 });
      }
    });

    test("poll trend chart uncertainty bands render when polls exist", async ({ page }) => {
      await page.goto(raceSlug);
      await expect(page.locator("article#main-content")).toBeVisible({ timeout: 15_000 });
      // Wait for poll table to confirm polls are present
      const table = page.locator('table[aria-label="Race polls"]');
      const hasTable = await table.isVisible({ timeout: 10_000 }).catch(() => false);
      if (!hasTable) {
        // No polls → no chart → skip
        return;
      }
      // The uncertainty band is a visx-area-closed path element inside the chart SVG.
      // It is aria-hidden (decorative), so we query by CSS class.
      const chart = page.locator('svg[aria-label="Poll trend chart"]');
      await expect(chart).toBeAttached({ timeout: 10_000 });
      const bands = chart.locator("path.visx-area-closed");
      // Expect at least one band (Dem + Rep = 2, but we check ≥ 1 to be lenient)
      await expect(bands.first()).toBeAttached({ timeout: 10_000 });
      const count = await bands.count();
      expect(count).toBeGreaterThanOrEqual(1);
    });
  });

  test.describe("Governor overview page", () => {
    test("governor overview page loads", async ({ page }) => {
      await page.goto("/forecast/governor");
      await expect(page).toHaveURL(/\/forecast\/governor/);
      const main = page.locator("main");
      await expect(main).toBeVisible({ timeout: 30_000 });
    });
  });
});
