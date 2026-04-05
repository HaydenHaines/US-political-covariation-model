/**
 * E2e tests for the CountySearch overlay on the stained-glass map.
 *
 * These tests navigate to the /explore/map page where MapShell renders and
 * verify that:
 *  1. The search input appears after the county GeoJSON loads
 *  2. Typing a query produces a dropdown of suggestions
 *  3. Selecting a suggestion clears the input and closes the dropdown
 *
 * NOTE: Map fly-to animation cannot be asserted in headless Playwright (deck.gl
 * renders to a canvas), so we only verify the UI state transitions.
 */

import { test, expect } from "@playwright/test";

const MAP_PAGE = "/explore/map";

/** Wait timeout for county GeoJSON to download (3.7MB, allow generous time) */
const GEO_TIMEOUT = 20_000;

test.describe("CountySearch overlay", () => {
  test("search input appears on the map page", async ({ page }) => {
    await page.goto(MAP_PAGE);
    // The search input is rendered once countyGeo loads
    const searchInput = page.getByRole("combobox", { name: /search for a county/i });
    await expect(searchInput).toBeVisible({ timeout: GEO_TIMEOUT });
  });

  test("typing opens a suggestion dropdown", async ({ page }) => {
    await page.goto(MAP_PAGE);
    const searchInput = page.getByRole("combobox", { name: /search for a county/i });
    await expect(searchInput).toBeVisible({ timeout: GEO_TIMEOUT });

    await searchInput.fill("Fulton");

    // Expect at least one suggestion to appear
    const listbox = page.getByRole("listbox", { name: /county suggestions/i });
    await expect(listbox).toBeVisible({ timeout: 5_000 });

    const options = page.getByRole("option");
    await expect(options.first()).toBeVisible();
    const firstText = await options.first().textContent();
    expect(firstText?.toLowerCase()).toContain("fulton");
  });

  test("selecting a suggestion clears the input and closes the dropdown", async ({ page }) => {
    await page.goto(MAP_PAGE);
    const searchInput = page.getByRole("combobox", { name: /search for a county/i });
    await expect(searchInput).toBeVisible({ timeout: GEO_TIMEOUT });

    await searchInput.fill("Fulton");

    const listbox = page.getByRole("listbox", { name: /county suggestions/i });
    await expect(listbox).toBeVisible({ timeout: 5_000 });

    // Click the first suggestion
    const firstOption = page.getByRole("option").first();
    await firstOption.click();

    // Input should be cleared after selection
    await expect(searchInput).toHaveValue("");

    // Dropdown should be gone
    await expect(listbox).not.toBeVisible({ timeout: 2_000 });
  });

  test("keyboard navigation: arrow keys move highlight, Enter selects", async ({ page }) => {
    await page.goto(MAP_PAGE);
    const searchInput = page.getByRole("combobox", { name: /search for a county/i });
    await expect(searchInput).toBeVisible({ timeout: GEO_TIMEOUT });

    await searchInput.fill("Los Angeles");

    const listbox = page.getByRole("listbox", { name: /county suggestions/i });
    await expect(listbox).toBeVisible({ timeout: 5_000 });

    // Move down to first option
    await searchInput.press("ArrowDown");
    // Press Enter to select
    await searchInput.press("Enter");

    // Input should be cleared after selection
    await expect(searchInput).toHaveValue("");
    await expect(listbox).not.toBeVisible({ timeout: 2_000 });
  });

  test("Escape key closes the dropdown without selecting", async ({ page }) => {
    await page.goto(MAP_PAGE);
    const searchInput = page.getByRole("combobox", { name: /search for a county/i });
    await expect(searchInput).toBeVisible({ timeout: GEO_TIMEOUT });

    await searchInput.fill("Cook");
    const listbox = page.getByRole("listbox", { name: /county suggestions/i });
    await expect(listbox).toBeVisible({ timeout: 5_000 });

    await searchInput.press("Escape");
    await expect(listbox).not.toBeVisible({ timeout: 2_000 });

    // Query text should remain in the input (Escape just closes the dropdown)
    await expect(searchInput).toHaveValue("Cook");
  });

  test("typing a query with no matches shows no dropdown", async ({ page }) => {
    await page.goto(MAP_PAGE);
    const searchInput = page.getByRole("combobox", { name: /search for a county/i });
    await expect(searchInput).toBeVisible({ timeout: GEO_TIMEOUT });

    await searchInput.fill("zzzzznotacountyname");

    // Dropdown should not appear for a non-matching query
    const listbox = page.getByRole("listbox", { name: /county suggestions/i });
    await expect(listbox).not.toBeVisible({ timeout: 2_000 });
  });
});
