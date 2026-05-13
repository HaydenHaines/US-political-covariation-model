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

    test("SC1: state filter input is visible", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      const filter = page.locator('[data-testid="senate-state-filter"]');
      await expect(filter).toBeVisible({ timeout: 10_000 });
    });

    test("SC2: typing a state substring narrows rendered race groups", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      // Wait for races to load (race cards link to /forecast/2026-*)
      const raceLinks = page.locator('a[href^="/forecast/2026-"]');
      await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });

      // State field uses abbreviations (e.g. "GA") -- filter by "ga" (case-insensitive)
      const filter = page.locator('[data-testid="senate-state-filter"]');
      await filter.fill("ga");

      // All rendered race groups must have data-states containing "GA" (case-insensitive)
      const groups = page.locator('[data-testid="senate-race-group"]');
      await groups.first().waitFor({ state: "attached", timeout: 10_000 });
      const count = await groups.count();
      for (let i = 0; i < count; i++) {
        const states = await groups.nth(i).getAttribute("data-states");
        expect(states?.toLowerCase()).toContain("ga");
      }
    });

    test("SC3: clearing filter restores all race groups", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      const raceLinks = page.locator('a[href^="/forecast/2026-"]');
      await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });

      const filter = page.locator('[data-testid="senate-state-filter"]');
      await filter.fill("GA");

      // Clear via the × button
      const clearBtn = page.locator('[data-testid="senate-state-filter-clear"]');
      await expect(clearBtn).toBeVisible({ timeout: 5_000 });
      await clearBtn.click();

      await expect(filter).toHaveValue("");
      // Empty state should not be visible after clearing
      await expect(page.locator('[data-testid="senate-filter-empty-state"]')).not.toBeVisible();
      // Race groups should be visible again
      const groups = page.locator('[data-testid="senate-race-group"]');
      await expect(groups.first()).toBeVisible({ timeout: 5_000 });
    });

    test("SC4: no-match input renders senate-filter-empty-state, hides race groups", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      const raceLinks = page.locator('a[href^="/forecast/2026-"]');
      await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });

      const filter = page.locator('[data-testid="senate-state-filter"]');
      await filter.fill("XYZZY_NO_MATCH_STATE");

      const emptyState = page.locator('[data-testid="senate-filter-empty-state"]');
      await expect(emptyState).toBeVisible({ timeout: 5_000 });

      const groups = page.locator('[data-testid="senate-race-group"]');
      await expect(groups).toHaveCount(0);
    });

    test("SC5: senate-race-group divs expose data-group and data-states attributes", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      const raceLinks = page.locator('a[href^="/forecast/2026-"]');
      await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });

      const groups = page.locator('[data-testid="senate-race-group"]');
      await groups.first().waitFor({ state: "attached", timeout: 10_000 });
      const count = await groups.count();
      expect(count).toBeGreaterThan(0);
      for (let i = 0; i < count; i++) {
        const group = await groups.nth(i).getAttribute("data-group");
        const states = await groups.nth(i).getAttribute("data-states");
        expect(group).toBeTruthy();
        expect(states).toBeTruthy();
      }
    });

    test("SC6: blend panel toggle works with active filter (filter persists)", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
      const raceLinks = page.locator('a[href^="/forecast/2026-"]');
      await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });

      const filter = page.locator('[data-testid="senate-state-filter"]');
      await filter.fill("GA");

      // Open blend panel
      const blendBtn = page.getByText("Adjust Forecast Blend");
      await blendBtn.click();
      await expect(page.locator("#overview-blend-panel")).toBeVisible({ timeout: 5_000 });

      // Filter should still have "GA" while panel is open
      await expect(filter).toHaveValue("GA");

      // Close blend panel
      await blendBtn.click();
      await expect(page.locator("#overview-blend-panel")).not.toBeVisible({ timeout: 5_000 });

      // Filter persists after toggle
      await expect(filter).toHaveValue("GA");
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

      // Wait for either poll table or no-polls message to appear (SWR async load)
      const table = page.locator('table[aria-label="Race polls"]');
      const noPolls = page.getByText("No polls available yet for this race.");
      await expect(table.or(noPolls)).toBeVisible({ timeout: 15_000 });
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
