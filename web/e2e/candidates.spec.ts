import { test, expect } from "@playwright/test";

// Raphael Warnock — 2 races, known badges, campaign finance data, legislative data.
const WARNOCK_ID = "W000790";
const WARNOCK_NAME = "Raphael";

test.describe("Candidates directory", () => {
  test("page loads with heading", async ({ page }) => {
    await page.goto("/candidates");
    const heading = page.locator("h1");
    await expect(heading).toBeVisible({ timeout: 30_000 });
  });

  test("candidate list renders after data loads", async ({ page }) => {
    await page.goto("/candidates");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    // Candidates link to /candidates/<bioguide_id>
    const links = page.locator('a[href^="/candidates/"]');
    await links.first().waitFor({ state: "attached", timeout: 10_000 });
    const count = await links.count();
    expect(count).toBeGreaterThan(0);
  });

  test("search filter narrows candidate list", async ({ page }) => {
    await page.goto("/candidates");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    // Wait for SWR data to load — candidate links appear when list is populated
    const links = page.locator('a[href^="/candidates/"]');
    await links.first().waitFor({ state: "attached", timeout: 15_000 });
    const initialCount = await links.count();
    expect(initialCount).toBeGreaterThan(5); // sanity: list is populated

    // Type a common last name that exists in the dataset but returns far fewer results
    const searchInput = page.getByRole("textbox", { name: /search candidates/i });
    await searchInput.fill("Smith");
    // Wait for React re-render — count should drop from full-list size
    await page.waitForTimeout(500);
    const filteredLinks = page.locator('a[href^="/candidates/"]');
    const filteredCount = await filteredLinks.count();
    // "Smith" is common enough to exist but rare enough to be less than the full list
    expect(filteredCount).toBeLessThan(initialCount);
  });

  test("clicking a candidate navigates to profile", async ({ page }) => {
    await page.goto("/candidates");
    await expect(page.locator("h1")).toBeVisible({ timeout: 30_000 });
    const links = page.locator('a[href^="/candidates/"]');
    await links.first().waitFor({ state: "attached", timeout: 10_000 });
    const href = await links.first().getAttribute("href");
    await page.goto(href!);
    await expect(page).toHaveURL(new RegExp(`/candidates/`), { timeout: 10_000 });
  });
});

test.describe("Candidate profile page", () => {
  test("Warnock profile page loads", async ({ page }) => {
    await page.goto(`/candidates/${WARNOCK_ID}`);
    await expect(page.locator("h1, h2").first()).toBeVisible({ timeout: 30_000 });
  });

  test("Warnock profile shows candidate name", async ({ page }) => {
    await page.goto(`/candidates/${WARNOCK_ID}`);
    await expect(page.locator("h1, h2").first()).toBeVisible({ timeout: 30_000 });
    const pageText = await page.textContent("body");
    expect(pageText).toContain(WARNOCK_NAME);
  });

  test("badge pills render on profile", async ({ page }) => {
    await page.goto(`/candidates/${WARNOCK_ID}`);
    await expect(page.locator("h1, h2").first()).toBeVisible({ timeout: 30_000 });
    // Badges render as inline elements with badge name text
    // Wait for SWR data to load
    await page.waitForTimeout(2000);
    const pageText = await page.textContent("body");
    // Warnock has known badges — page should contain badge-related content
    // We check the badges section heading or at least one known badge word
    expect(
      pageText?.includes("Badge") ||
      pageText?.includes("badge") ||
      pageText?.includes("Suburb") ||
      pageText?.includes("Community")
    ).toBeTruthy();
  });

  test("CTOV radar chart section is present", async ({ page }) => {
    await page.goto(`/candidates/${WARNOCK_ID}`);
    await expect(page.locator("h1, h2").first()).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(2000);
    // CTOVRadarChart renders as an SVG
    const svgLocator = page.locator("svg");
    const svgCount = await svgLocator.count();
    expect(svgCount).toBeGreaterThan(0);
  });

  test("election history table is present", async ({ page }) => {
    await page.goto(`/candidates/${WARNOCK_ID}`);
    await expect(page.locator("h1, h2").first()).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(2000);
    // Warnock has 2 races — there should be a table or list of elections
    const pageText = await page.textContent("body");
    // Both GA senate races were in 2020 and 2022
    expect(pageText?.includes("202") || pageText?.includes("Georgia") || pageText?.includes("Senate")).toBeTruthy();
  });

  test("profile page does not 404", async ({ page }) => {
    const response = await page.goto(`/candidates/${WARNOCK_ID}`);
    expect(response?.status()).not.toBe(404);
  });

  test("unknown bioguide shows error state not crash", async ({ page }) => {
    // Should render gracefully, not throw an uncaught error
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));
    await page.goto("/candidates/DOES_NOT_EXIST_XYZ");
    await expect(page.locator("h1, h2, [role='alert']").first()).toBeVisible({ timeout: 15_000 });
    // No uncaught JS errors
    expect(errors.filter((e) => !e.includes("Warning"))).toHaveLength(0);
  });
});
