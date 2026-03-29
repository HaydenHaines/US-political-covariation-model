import { test, expect } from "@playwright/test";

test.describe("Landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("hero section renders with headline text", async ({ page }) => {
    // The landing page uses useSenateOverview — wait for data or loading state
    const hero = page.locator("section").first();
    await expect(hero).toBeVisible({ timeout: 10_000 });

    // Either the headline (h1) or a loading skeleton should be present
    const headlineOrSkeleton = page.locator("h1, [data-slot='skeleton']").first();
    await expect(headlineOrSkeleton).toBeVisible({ timeout: 10_000 });
  });

  test("hero h1 appears once data loads", async ({ page }) => {
    // Wait for the h1 which renders once API data arrives
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 15_000 });
    // h1 should have non-empty text
    const text = await h1.textContent();
    expect(text?.trim().length).toBeGreaterThan(0);
  });

  test("race ticker section is present", async ({ page }) => {
    // The "Closest Races" heading is always rendered (loading or data state)
    const tickerHeading = page.getByText("Closest Races");
    await expect(tickerHeading).toBeVisible({ timeout: 10_000 });
  });

  test("race ticker items are clickable links once loaded", async ({ page }) => {
    // Wait for ticker items (links inside the ticker scroll strip)
    // Ticker links navigate to /forecast/[slug]
    const tickerLinks = page.locator('a[href^="/forecast/"]');
    await tickerLinks.first().waitFor({ state: "visible", timeout: 15_000 });
    const count = await tickerLinks.count();
    expect(count).toBeGreaterThan(0);
    // Each ticker link should be a valid anchor
    const firstHref = await tickerLinks.first().getAttribute("href");
    expect(firstHref).toMatch(/^\/forecast\/.+/);
  });

  test("entry point cards link to correct pages", async ({ page }) => {
    // Three entry point cards: /forecast, /types, /methodology
    const forecastLink = page.locator('a[href="/forecast"]');
    const typesLink = page.locator('a[href="/types"]');
    const methodologyLink = page.locator('a[href="/methodology"]');

    await expect(forecastLink).toBeVisible({ timeout: 10_000 });
    await expect(typesLink).toBeVisible();
    await expect(methodologyLink).toBeVisible();
  });

  test("footer renders with navigation links", async ({ page }) => {
    const footer = page.locator("footer");
    await expect(footer).toBeVisible({ timeout: 10_000 });
    // Footer contains nav links
    const footerNav = footer.locator("nav");
    await expect(footerNav).toBeVisible();
    const links = footerNav.locator("a");
    const linkCount = await links.count();
    expect(linkCount).toBeGreaterThan(0);
  });

  test("footer contains WetherVane branding text", async ({ page }) => {
    const footer = page.locator("footer");
    await expect(footer).toBeVisible({ timeout: 10_000 });
    await expect(footer).toContainText("WetherVane");
  });
});
