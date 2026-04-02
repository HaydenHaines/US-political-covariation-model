import { test, expect } from "@playwright/test";

test("landing page loads and shows headline", async ({ page }) => {
  await page.goto("/");
  const main = page.locator("main");
  await expect(main).toBeVisible({ timeout: 10_000 });
  const heading = page.locator("h1");
  await expect(heading).toBeVisible({ timeout: 10_000 });
});

test("landing page has navigation to forecast", async ({ page }) => {
  await page.goto("/");
  const forecastLink = page.locator('a[href="/forecast"]');
  await expect(forecastLink.first()).toBeVisible({ timeout: 10_000 });
});

test("forecast senate page loads race data", async ({ page }) => {
  await page.goto("/forecast/senate");
  const h1 = page.locator("h1");
  await expect(h1).toBeVisible({ timeout: 30_000 });
  const text = await h1.textContent();
  expect(text).toContain("Senate");
});

test("forecast senate page has race cards in DOM", async ({ page }) => {
  await page.goto("/forecast/senate");
  await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
  // Race cards may be below the fold — check attached state
  const raceLinks = page.locator('a[href^="/forecast/2026-"]');
  await raceLinks.first().waitFor({ state: "attached", timeout: 10_000 });
  const count = await raceLinks.count();
  expect(count).toBeGreaterThan(0);
});
