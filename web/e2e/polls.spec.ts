import { test, expect } from "@playwright/test";

test.describe("Polls page", () => {
  test("page loads with table and shows poll count", async ({ page }) => {
    await page.goto("/polls");
    const table = page.locator('[data-testid="polls-table"]');
    await expect(table).toBeVisible({ timeout: 15_000 });

    // Poll count indicator should be present and show a number
    const countEl = page.locator('[data-testid="poll-count"]');
    await expect(countEl).toBeVisible();
    const countText = await countEl.textContent();
    expect(countText).toMatch(/\d+ polls?/);
  });

  test("sort by clicking a column header changes order", async ({ page }) => {
    await page.goto("/polls");
    const table = page.locator('[data-testid="polls-table"]');
    await expect(table).toBeVisible({ timeout: 15_000 });

    // Capture first row text before sorting
    const firstRowBefore = await table.locator("tbody tr").first().textContent();

    // Click "Race" header to sort by race ascending
    await page.locator("th", { hasText: "Race" }).click();

    // First row should change (date desc → race asc)
    const firstRowAfter = await table.locator("tbody tr").first().textContent();
    expect(firstRowAfter).not.toBe(firstRowBefore);
  });

  test("filter by race dropdown reduces row count", async ({ page }) => {
    await page.goto("/polls");
    const countEl = page.locator('[data-testid="poll-count"]');
    await expect(countEl).toBeVisible({ timeout: 15_000 });

    // Get initial count
    const initialText = (await countEl.textContent()) ?? "";
    const initialCount = parseInt(initialText.match(/(\d+)/)?.[1] ?? "0", 10);
    expect(initialCount).toBeGreaterThan(0);

    // Select the first non-"all" option in the race filter
    const raceFilter = page.locator('[data-testid="race-filter"]');
    const options = await raceFilter.locator("option").allTextContents();
    // options[0] is "All Races", pick options[1]
    expect(options.length).toBeGreaterThan(1);
    await raceFilter.selectOption({ index: 1 });

    // Count should decrease (or stay same if all polls are that race, but unlikely)
    const filteredText = (await countEl.textContent()) ?? "";
    const filteredCount = parseInt(filteredText.match(/(\d+)/)?.[1] ?? "0", 10);
    expect(filteredCount).toBeLessThan(initialCount);
    expect(filteredText).toContain("(filtered)");
  });

  test("breadcrumb navigation works", async ({ page }) => {
    await page.goto("/polls");
    const breadcrumb = page.locator('nav[aria-label="breadcrumb"]');
    await expect(breadcrumb).toBeVisible({ timeout: 15_000 });

    // Should show "Home / Polls"
    const text = await breadcrumb.textContent();
    expect(text).toContain("Home");
    expect(text).toContain("Polls");

    // Click Home link navigates away
    await breadcrumb.locator("a", { hasText: "Home" }).click();
    await expect(page).toHaveURL("/");
  });

  test("no console errors on polls page", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    await page.goto("/polls");
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
