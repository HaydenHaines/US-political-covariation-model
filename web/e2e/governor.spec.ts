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
});
