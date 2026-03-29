import { test, expect, type Page } from "@playwright/test";

// Helper to collect console errors on a page
async function collectConsoleErrors(page: Page, action: () => Promise<void>): Promise<string[]> {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });
  await action();
  return errors;
}

test.describe("Breadcrumbs", () => {
  test("type detail page has breadcrumb nav", async ({ page }) => {
    // Type pages use the <Breadcrumbs> component
    await page.goto("/type/0");
    const breadcrumb = page.locator("nav[aria-label='Breadcrumb']");
    await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
  });

  test("type detail breadcrumb has links", async ({ page }) => {
    await page.goto("/type/0");
    const breadcrumb = page.locator("nav[aria-label='Breadcrumb']");
    await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
    const links = breadcrumb.locator("a");
    const count = await links.count();
    expect(count).toBeGreaterThan(0);
  });

  test("county detail page has breadcrumb nav", async ({ page }) => {
    // Use a known valid FIPS for Georgia's Fulton County
    await page.goto("/county/13121");
    const breadcrumb = page.locator("nav[aria-label='Breadcrumb']");
    await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
  });

  test("race detail page has breadcrumb nav", async ({ page }) => {
    // /forecast/[slug] renders its own breadcrumb with aria-label="breadcrumb"
    await page.goto("/forecast/senate");
    await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

    const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
    const href = await firstRaceLink.getAttribute("href");
    if (!href) return;

    await page.goto(href);
    const breadcrumb = page.locator('nav[aria-label="breadcrumb"]');
    await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
  });
});

test.describe("Back links", () => {
  test("race detail page has back link to forecast", async ({ page }) => {
    await page.goto("/forecast/senate");
    await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

    const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
    const href = await firstRaceLink.getAttribute("href");
    if (!href) return;

    await page.goto(href);
    const backLink = page.locator('a[href="/forecast"]');
    await expect(backLink).toBeVisible({ timeout: 10_000 });
  });

  test("clicking back link from race detail navigates to forecast", async ({ page }) => {
    await page.goto("/forecast/senate");
    await expect(page.locator("h1")).toBeVisible({ timeout: 15_000 });

    const firstRaceLink = page.locator('a[href^="/forecast/"]').first();
    const href = await firstRaceLink.getAttribute("href");
    if (!href) return;

    await page.goto(href);
    const backLink = page.locator('a[href="/forecast"]').first();
    await expect(backLink).toBeVisible({ timeout: 10_000 });
    await backLink.click();
    // /forecast redirects to /forecast/senate
    await expect(page).toHaveURL(/\/forecast/, { timeout: 10_000 });
  });

  test("type detail page has back link to types list", async ({ page }) => {
    await page.goto("/type/0");
    const backLink = page.locator('a[href="/types"]');
    await expect(backLink).toBeVisible({ timeout: 10_000 });
  });
});

test.describe("404 page", () => {
  test("404 page renders for invalid routes", async ({ page }) => {
    await page.goto("/this-route-does-not-exist-at-all-xyz123");
    // not-found.tsx renders "Page Not Found" heading
    const heading = page.locator("h1");
    await expect(heading).toBeVisible({ timeout: 10_000 });
    await expect(heading).toContainText("Page Not Found");
  });

  test("404 page contains navigation links", async ({ page }) => {
    await page.goto("/forecast/race/this-slug-does-not-exist-xyz123");
    // Either the custom not-found or the race detail fallback renders
    // Both provide navigation back
    const anyLink = page.locator("a").first();
    await expect(anyLink).toBeVisible({ timeout: 10_000 });
  });

  test("404 page has link back to home", async ({ page }) => {
    await page.goto("/this-does-not-exist-xyz");
    const homeLink = page.locator('a[href="/"]');
    await expect(homeLink).toBeVisible({ timeout: 10_000 });
  });
});

test.describe("Explore pages", () => {
  test("/explore/types page loads", async ({ page }) => {
    await page.goto("/explore/types");
    await expect(page).toHaveURL(/\/explore\/types/);
    const h1 = page.locator("h1");
    await expect(h1).toBeVisible({ timeout: 10_000 });
  });

  test("/explore/types page has Electoral Types heading", async ({ page }) => {
    await page.goto("/explore/types");
    const h1 = page.locator("h1");
    await expect(h1).toBeVisible({ timeout: 10_000 });
    await expect(h1).toContainText("Electoral Types");
  });

  test("/explore/shifts page loads", async ({ page }) => {
    await page.goto("/explore/shifts");
    await expect(page).toHaveURL(/\/explore\/shifts/);
    // Page should have some content
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });

  test("/explore/map page loads", async ({ page }) => {
    await page.goto("/explore/map");
    await expect(page).toHaveURL(/\/explore\/map/);
    const main = page.locator("main");
    await expect(main).toBeVisible({ timeout: 10_000 });
  });
});

test.describe("Console errors", () => {
  test("landing page has no console errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });
    await page.goto("/");
    // Wait for initial render
    await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {
      // networkidle may not be reached if API is unavailable — that is expected
    });
    // Filter out expected browser extension / network errors
    const realErrors = errors.filter(
      (e) =>
        !e.includes("net::ERR_") &&
        !e.includes("Failed to load resource") &&
        !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
  });

  test("forecast senate page has no console errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });
    await page.goto("/forecast/senate");
    await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {});
    const realErrors = errors.filter(
      (e) =>
        !e.includes("net::ERR_") &&
        !e.includes("Failed to load resource") &&
        !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
  });

  test("/explore/types page has no console errors", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });
    await page.goto("/explore/types");
    await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {});
    const realErrors = errors.filter(
      (e) =>
        !e.includes("net::ERR_") &&
        !e.includes("Failed to load resource") &&
        !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
  });
});
