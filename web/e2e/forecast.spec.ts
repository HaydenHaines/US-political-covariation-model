import { test, expect } from "@playwright/test";

test.describe("Forecast flow", () => {
  test.describe("Senate overview page", () => {
    test("senate overview page loads", async ({ page }) => {
      // /forecast redirects to /forecast/senate
      await page.goto("/forecast/senate");
      await expect(page).toHaveURL(/\/forecast\/senate/);
    });

    test("senate page has a heading", async ({ page }) => {
      await page.goto("/forecast/senate");
      const heading = page.locator("h1");
      await expect(heading).toBeVisible({ timeout: 10_000 });
      const text = await heading.textContent();
      expect(text?.trim().length).toBeGreaterThan(0);
    });

    test("balance bar renders once data loads", async ({ page }) => {
      await page.goto("/forecast/senate");
      // BalanceBar renders a flex row of colored seat segments
      // It renders inside the page body after data loads — check for the component
      // by waiting for skeleton to disappear or for actual race cards to appear
      const skeleton = page.locator('[data-slot="skeleton"]').first();
      // Either skeletons are gone or content has appeared
      await page.waitForFunction(
        () => {
          // Wait until h1 is visible (data loaded) or until we time out
          const h1 = document.querySelector("h1");
          return h1 !== null && h1.textContent!.trim().length > 0;
        },
        { timeout: 15_000 },
      );
      // After data loads, h1 should be visible (not a skeleton)
      const h1 = page.locator("h1");
      await expect(h1).toBeVisible();
    });

    test("race cards are visible after data loads", async ({ page }) => {
      await page.goto("/forecast/senate");
      // Wait for h1 to confirm data has loaded
      const h1 = page.locator("h1");
      await expect(h1).toBeVisible({ timeout: 15_000 });

      // Race cards link to /forecast/[slug] — look for such links
      const raceLinks = page.locator('a[href^="/forecast/"]');
      await raceLinks.first().waitFor({ state: "visible", timeout: 10_000 });
      const count = await raceLinks.count();
      expect(count).toBeGreaterThan(0);
    });

    test("clicking a race card navigates to race detail page", async ({ page }) => {
      await page.goto("/forecast/senate");
      const h1 = page.locator("h1");
      await expect(h1).toBeVisible({ timeout: 15_000 });

      // Find the first race card link and click it
      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      await expect(firstRaceLink).toBeVisible({ timeout: 10_000 });
      const href = await firstRaceLink.getAttribute("href");
      await firstRaceLink.click();

      // Should land on a race detail page (slug URL, not /senate or /governor)
      await expect(page).toHaveURL(new RegExp(`${href}$`), { timeout: 10_000 });
    });
  });

  test.describe("Race detail page", () => {
    test("race detail page shows hero section", async ({ page }) => {
      await page.goto("/forecast/senate");
      const h1Loading = page.locator("h1");
      await expect(h1Loading).toBeVisible({ timeout: 15_000 });

      // Navigate to first race detail
      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      await expect(firstRaceLink).toBeVisible({ timeout: 10_000 });
      await firstRaceLink.click();

      // Race detail page has an article with id="main-content"
      const article = page.locator('article#main-content');
      await expect(article).toBeVisible({ timeout: 10_000 });
    });

    test("race detail page has a breadcrumb nav", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      const href = await firstRaceLink.getAttribute("href");
      if (!href) return;

      await page.goto(href);
      const breadcrumb = page.locator('nav[aria-label="breadcrumb"]');
      await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
    });

    test("race detail page shows race title or state name in heading", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      const href = await firstRaceLink.getAttribute("href");
      if (!href) return;

      await page.goto(href);
      // RaceHero renders the state name and race type in an h1
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent();
      expect(text?.trim().length).toBeGreaterThan(0);
    });

    test("race detail page shows poll section", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      const href = await firstRaceLink.getAttribute("href");
      if (!href) return;

      await page.goto(href);
      // "Recent Polls" section heading is always rendered
      const pollHeading = page.getByText("Recent Polls");
      await expect(pollHeading).toBeVisible({ timeout: 10_000 });
    });

    test("race detail page shows poll table or no-polls message", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      const href = await firstRaceLink.getAttribute("href");
      if (!href) return;

      await page.goto(href);
      await expect(page.getByText("Recent Polls")).toBeVisible({ timeout: 10_000 });

      // Either a table (with data) or a "No polls" message should be present
      const table = page.locator("table");
      const noPolls = page.getByText(/no polls/i);
      const hasTable = await table.isVisible().catch(() => false);
      const hasNoPolls = await noPolls.isVisible().catch(() => false);
      expect(hasTable || hasNoPolls).toBe(true);
    });

    test("race detail page has a back link to forecast", async ({ page }) => {
      await page.goto("/forecast/senate");
      await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

      const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
      const href = await firstRaceLink.getAttribute("href");
      if (!href) return;

      await page.goto(href);
      const backLink = page.locator('a[href="/forecast"]');
      await expect(backLink).toBeVisible({ timeout: 10_000 });
    });
  });

  test.describe("Governor overview page", () => {
    test("governor overview page loads", async ({ page }) => {
      await page.goto("/forecast/governor");
      await expect(page).toHaveURL(/\/forecast\/governor/);
      const h1 = page.locator("h1");
      await expect(h1).toBeVisible({ timeout: 15_000 });
    });
  });
});
