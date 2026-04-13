import { test, expect } from "@playwright/test";

test.describe("Landing page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("page renders with main content", async ({ page }) => {
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("senate headline appears", async ({ page }) => {
    const heading = page.locator("h1");
    await expect(heading).toBeVisible({ timeout: 10_000 });
    const text = await heading.textContent();
    expect(text).toContain("Senate");
  });

  test("scrollytelling zones are present", async ({ page }) => {
    // The homepage has scrollytelling zones with narrative text
    await expect(page.getByText("Scroll to explore")).toBeVisible({ timeout: 10_000 });
  });

  test("navigation links to key sections", async ({ page }) => {
    const forecastLink = page.locator('a[href="/forecast"]');
    const exploreLink = page.locator('a[href="/types"]');
    const methodologyLink = page.locator('a[href="/methodology"]');

    await expect(forecastLink.first()).toBeVisible({ timeout: 10_000 });
    await expect(exploreLink.first()).toBeVisible();
    await expect(methodologyLink.first()).toBeVisible();
  });

  test("senate control summary is present", async ({ page }) => {
    // Homepage shows senate seat balance (e.g., "33D not up" / "34R not up")
    await expect(page.getByText(/not up/i).first()).toBeVisible({ timeout: 10_000 });
  });

  test("footer renders with navigation links", async ({ page }) => {
    const footer = page.locator("footer");
    await expect(footer).toBeVisible({ timeout: 10_000 });
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
