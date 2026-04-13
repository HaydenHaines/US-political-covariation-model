import { test, expect } from "@playwright/test";

test.describe("Methodology pages", () => {
  test.describe("/methodology", () => {
    test("page loads with main content", async ({ page }) => {
      await page.goto("/methodology");
      // Two <main id="main-content"> exist (outer layout + page content) — use h1 as the load signal
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
    });

    test("page has a heading containing 'Methodology'", async ({ page }) => {
      await page.goto("/methodology");
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent();
      expect(text).toBeTruthy();
    });

    test("Key Insight section is present", async ({ page }) => {
      await page.goto("/methodology");
      await page.locator("h1").first().waitFor({ timeout: 10_000 });
      await expect(page.getByRole("heading", { name: "The Key Insight" })).toBeVisible({ timeout: 10_000 });
    });

    test("How It Works section is present", async ({ page }) => {
      await page.goto("/methodology");
      await page.locator("h1").first().waitFor({ timeout: 10_000 });
      await expect(page.getByRole("heading", { name: "How It Works" })).toBeVisible({ timeout: 10_000 });
    });

    test("Model Performance section is present", async ({ page }) => {
      await page.goto("/methodology");
      await page.locator("h1").first().waitFor({ timeout: 10_000 });
      await expect(page.getByRole("heading", { name: "Model Performance" })).toBeVisible({ timeout: 10_000 });
    });

    test("has a link to accuracy page", async ({ page }) => {
      await page.goto("/methodology");
      await page.locator("h1").first().waitFor({ timeout: 10_000 });
      const accuracyLink = page.locator('a[href*="accuracy"]');
      await accuracyLink.first().waitFor({ state: "attached", timeout: 10_000 });
      const count = await accuracyLink.count();
      expect(count).toBeGreaterThan(0);
    });

    test("no console errors on methodology page", async ({ page }) => {
      const errors: string[] = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") errors.push(msg.text());
      });
      await page.goto("/methodology");
      await page.locator("h1").first().waitFor({ timeout: 10_000 });
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

  test.describe("/methodology/accuracy", () => {
    test("page loads with main content", async ({ page }) => {
      await page.goto("/methodology/accuracy");
      const article = page.locator("article#main-content");
      await expect(article).toBeVisible({ timeout: 10_000 });
    });

    test("page has a heading with 'Model Accuracy' or similar", async ({ page }) => {
      await page.goto("/methodology/accuracy");
      const h1 = page.locator("h1").first();
      await expect(h1).toBeVisible({ timeout: 10_000 });
      const text = await h1.textContent();
      expect(text).toBeTruthy();
    });

    test("metrics grid is present with LOO score", async ({ page }) => {
      await page.goto("/methodology/accuracy");
      await expect(page.locator("article#main-content")).toBeVisible({ timeout: 10_000 });
      // MetricCard renders the LOO r value as a heading metric — use first() to avoid strict mode
      await expect(page.getByText(/0\.731/).first()).toBeVisible({ timeout: 10_000 });
    });

    test("Overall Performance section is present", async ({ page }) => {
      await page.goto("/methodology/accuracy");
      await expect(page.locator("article#main-content")).toBeVisible({ timeout: 10_000 });
      await expect(page.getByRole("heading", { name: "Overall Performance" })).toBeVisible({ timeout: 10_000 });
    });

    test("Cross-Election Validation section is present", async ({ page }) => {
      await page.goto("/methodology/accuracy");
      await expect(page.locator("article#main-content")).toBeVisible({ timeout: 10_000 });
      await expect(page.getByRole("heading", { name: "Cross-Election Validation" })).toBeVisible({ timeout: 10_000 });
    });

    test("back link to methodology page is present", async ({ page }) => {
      await page.goto("/methodology/accuracy");
      await expect(page.locator("article#main-content")).toBeVisible({ timeout: 10_000 });
      const backLink = page.locator('a[href="/methodology"]');
      await expect(backLink.first()).toBeVisible({ timeout: 10_000 });
    });

    test("no console errors on accuracy page", async ({ page }) => {
      const errors: string[] = [];
      page.on("console", (msg) => {
        if (msg.type() === "error") errors.push(msg.text());
      });
      await page.goto("/methodology/accuracy");
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
});
