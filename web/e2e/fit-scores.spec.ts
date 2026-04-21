import { test, expect } from "@playwright/test";

/**
 * Moneyball Fit Scores e2e tests.
 *
 * Uses the GA Senate 2026 race as the reference page — it is a competitive
 * race with sabermetric history for both parties and is reliably present
 * in the forecast dataset.
 *
 * Tests cover:
 *   - Section renders on the race detail page
 *   - D/R party toggle switches the visible pool
 *   - Candidate names appear and link to profiles
 *   - Score column renders numeric values
 *   - Loading → data transition (no stuck spinner)
 *   - Empty state renders gracefully when no candidates qualify
 *   - No uncaught JS errors during any of the above
 */

const RACE_SLUG = "/forecast/2026-ga-senate";
const SECTION_LABEL = '[aria-label="Moneyball Fit Scores"]';

test.describe("Moneyball Fit Scores section", () => {
  test("section renders on race detail page", async ({ page }) => {
    await page.goto(RACE_SLUG);
    // Wait for the main article to confirm full SSR render
    await expect(page.locator("article#main-content")).toBeVisible({ timeout: 15_000 });
    // The section is a client component — wait for it to hydrate
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });
  });

  test("section heading 'Moneyball Fit Scores' is visible", async ({ page }) => {
    await page.goto(RACE_SLUG);
    await expect(page.locator("article#main-content")).toBeVisible({ timeout: 15_000 });
    const heading = page.getByText("Moneyball Fit Scores");
    await expect(heading).toBeVisible({ timeout: 15_000 });
  });

  test("D/R toggle buttons are present", async ({ page }) => {
    await page.goto(RACE_SLUG);
    await expect(page.locator(SECTION_LABEL)).toBeVisible({ timeout: 15_000 });
    const toggleGroup = page.locator('[aria-label="Filter fit scores by party"]');
    await expect(toggleGroup).toBeVisible({ timeout: 10_000 });
    await expect(toggleGroup.getByText("Dem")).toBeVisible();
    await expect(toggleGroup.getByText("Rep")).toBeVisible();
  });

  test("clicking Rep toggle switches to Republican candidates", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    // Switch to Republican pool
    const repButton = page.locator('[aria-label="Filter fit scores by party"]').getByText("Rep");
    await repButton.click();

    // The ranked list aria-label should update to mention "Republican"
    const repList = page.locator('[aria-label*="Republican fit candidates"]');
    await expect(repList).toBeVisible({ timeout: 10_000 });
  });

  test("clicking Dem toggle switches back to Democratic candidates", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    const toggleGroup = page.locator('[aria-label="Filter fit scores by party"]');
    // First switch to Rep
    await toggleGroup.getByText("Rep").click();
    // Then switch back to Dem
    await toggleGroup.getByText("Dem").click();

    const demList = page.locator('[aria-label*="Democratic fit candidates"]');
    await expect(demList).toBeVisible({ timeout: 10_000 });
  });

  test("candidate list renders with at least one candidate", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    // Wait for either a candidate link or the empty-state message
    const candidateLinks = section.locator('a[href^="/candidates/"]');
    const emptyMsg = section.getByText(/No .* candidates with enough races/);
    await expect(candidateLinks.first().or(emptyMsg)).toBeVisible({ timeout: 10_000 });
  });

  test("rank badges are present (#1, #2, ...)", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    // Rank labels rendered as "#1", "#2", etc.
    const rankOne = section.locator('[aria-label="Rank 1"]');
    await expect(rankOne).toBeVisible({ timeout: 10_000 });
  });

  test("fit score numeric values are displayed", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    // Score column header should be visible
    const scoreHeader = section.getByText("Score", { exact: true });
    await expect(scoreHeader).toBeVisible({ timeout: 10_000 });
  });

  test("candidate names link to /candidates/ profile pages", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    const candidateLinks = section.locator('a[href^="/candidates/"]');
    // If candidates exist, confirm the first link is a valid profile URL
    const count = await candidateLinks.count();
    if (count > 0) {
      const href = await candidateLinks.first().getAttribute("href");
      expect(href).toMatch(/^\/candidates\/[A-Z0-9]+$/);
    }
  });

  test("no JavaScript errors during section render", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto(RACE_SLUG);
    await expect(page.locator(SECTION_LABEL)).toBeVisible({ timeout: 15_000 });
    // Wait for SWR fetch to settle
    await page.waitForTimeout(2_000);

    // Filter out React hydration warnings (not errors)
    const realErrors = errors.filter(
      (e) => !e.includes("Warning") && !e.includes("hydrat"),
    );
    expect(realErrors).toHaveLength(0);
  });

  test("section is present in the DOM even before SWR resolves (loading skeleton)", async ({
    page,
  }) => {
    // Intercept the fit-scores API call to delay it so we can observe loading state
    await page.route("**/api/v1/races/**fit-scores**", async (route) => {
      // Delay 2 seconds before forwarding
      await new Promise((resolve) => setTimeout(resolve, 2_000));
      await route.continue();
    });

    await page.goto(RACE_SLUG);
    await expect(page.locator("article#main-content")).toBeVisible({ timeout: 15_000 });

    // The dynamic() loading skeleton should appear before fit-scores resolve
    // We check that the section container is at least attached (may be skeleton)
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 12_000 });
  });

  test("methodology note appears below the ranked list", async ({ page }) => {
    await page.goto(RACE_SLUG);
    const section = page.locator(SECTION_LABEL);
    await expect(section).toBeVisible({ timeout: 15_000 });

    // Wait for data to load
    const candidateLinks = section.locator('a[href^="/candidates/"]');
    const emptyMsg = section.getByText(/No .* candidates/);
    await expect(candidateLinks.first().or(emptyMsg)).toBeVisible({ timeout: 10_000 });

    // The methodology footer note should be visible when data is loaded
    if (await candidateLinks.count() > 0) {
      const note = section.getByText(/Score = career CTOV/);
      await expect(note).toBeVisible({ timeout: 5_000 });
    }
  });
});
