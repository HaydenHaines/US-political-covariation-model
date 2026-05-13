import { test, expect } from "@playwright/test";

test.describe("Governor overview page", () => {
  test("governor page loads and shows heading", async ({ page }) => {
    await page.goto("/forecast/governor");
    const heading = page.locator("h1");
    await expect(heading).toBeVisible({ timeout: 30_000 });
    const text = await heading.textContent();
    expect(text).toContain("Governor");
  });

  test("governor page shows race cards", async ({ page }) => {
    await page.goto("/forecast/governor");
    // Wait for the heading so we know the page has hydrated
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

    // Race cards are links to /forecast/2026-xx-governor
    const raceLinks = page.locator('a[href*="-governor"]');
    await raceLinks.first().waitFor({ state: "attached", timeout: 15_000 });
    const count = await raceLinks.count();
    expect(count).toBeGreaterThan(0);
  });

  test("race card links point to governor detail pages", async ({ page }) => {
    await page.goto("/forecast/governor");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });

    const firstLink = page.locator('a[href*="-governor"]').first();
    await firstLink.waitFor({ state: "attached", timeout: 15_000 });
    const href = await firstLink.getAttribute("href");
    expect(href).toMatch(/\/forecast\/2026-[a-z]+-governor/);
  });

  // SC1: state-filter input is visible between header <p> and FundamentalsCard
  test("state filter input is visible on page load", async ({ page }) => {
    await page.goto("/forecast/governor");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    await expect(page.locator('[data-testid="state-filter"]')).toBeVisible();
  });

  // SC2: typing filters race cards case-insensitively (state field is 2-letter abbrev: "OH")
  test("typing a state abbreviation filters race cards", async ({ page }) => {
    await page.goto("/forecast/governor");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    // Wait for race cards to load first
    const raceLinks = page.locator('a[href*="-governor"]');
    await raceLinks.first().waitFor({ state: "attached", timeout: 15_000 });
    const totalBefore = await raceLinks.count();
    expect(totalBefore).toBeGreaterThan(1);

    // "OH" matches only Ohio — case-insensitive substring on the 2-letter state code.
    // RaceCardGrid renders each card twice (mobile carousel + desktop grid), so 1 race = 2 links.
    await page.locator('[data-testid="state-filter"]').fill("OH");
    await expect(raceLinks).toHaveCount(2, { timeout: 5_000 });

    // "oh" lowercase should also match (case-insensitive)
    await page.locator('[data-testid="state-filter"]').fill("oh");
    await expect(raceLinks).toHaveCount(2, { timeout: 5_000 });
  });

  // SC3: clearing the filter restores all cards
  test("clearing state filter restores all race cards", async ({ page }) => {
    await page.goto("/forecast/governor");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    const raceLinks = page.locator('a[href*="-governor"]');
    await raceLinks.first().waitFor({ state: "attached", timeout: 15_000 });
    const totalBefore = await raceLinks.count();

    await page.locator('[data-testid="state-filter"]').fill("OH");
    await expect(raceLinks).toHaveCount(2, { timeout: 5_000 });

    await page.locator('[data-testid="state-filter"]').fill("");
    await expect(raceLinks).toHaveCount(totalBefore, { timeout: 5_000 });
  });

  // SC4: no-match input shows empty-state message; FundamentalsCard stays mounted
  test("no-match filter shows empty state and keeps FundamentalsCard", async ({ page }) => {
    await page.goto("/forecast/governor");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    const raceLinks = page.locator('a[href*="-governor"]');
    await raceLinks.first().waitFor({ state: "attached", timeout: 15_000 });

    await page.locator('[data-testid="state-filter"]').fill("ZZZNOMATCH");
    // Race cards should be gone
    await expect(raceLinks).toHaveCount(0, { timeout: 5_000 });
    // Empty state message appears
    await expect(page.locator("text=No governor races match")).toBeVisible();
    // FundamentalsCard stays mounted (aria-label="National Environment")
    await expect(page.locator('[aria-label="National Environment"]')).toBeVisible();
  });
});
