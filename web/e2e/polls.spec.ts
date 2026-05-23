import { test, expect, type Page } from "@playwright/test";

const MOCK_POLLS = [
  {
    race: "WI-GOV",
    geography: "Wisconsin",
    geo_level: "state",
    dem_share: 0.48,
    n_sample: 900,
    date: "2026-03-11",
    pollster: "North Star",
    grade: "B+",
  },
  {
    race: "AZ-SEN",
    geography: "Arizona",
    geo_level: "state",
    dem_share: 0.51,
    n_sample: 1100,
    date: "2026-04-18",
    pollster: "Desert Research",
    grade: "C",
  },
  {
    race: "GA-GOV",
    geography: "Georgia",
    geo_level: "state",
    dem_share: 0.46,
    n_sample: 775,
    date: "2026-01-24",
    pollster: "Peach State Polling",
    grade: "A",
  },
  {
    race: "PA-SEN",
    geography: "Pennsylvania",
    geo_level: "state",
    dem_share: 0.5,
    n_sample: 1250,
    date: "2026-02-02",
    pollster: "Keystone Analytics",
    grade: "B-",
  },
];

const GRADE_ORDER: Record<string, number> = {
  A: 1,
  "A-": 2,
  "B+": 3,
  B: 4,
  "B-": 5,
  "C+": 6,
  C: 7,
  "C-": 8,
  "—": 99,
};

async function mockPollsData(page: Page) {
  await page.route("**/api/v1/polls", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(MOCK_POLLS),
    });
  });
}

async function gotoPolls(page: Page, query = "") {
  await mockPollsData(page);
  await page.goto(`/polls${query}`);
  await expect(page.locator('[data-testid="polls-table"]')).toBeVisible({
    timeout: 15_000,
  });
}

async function columnTexts(page: Page, columnIndex: number) {
  return page
    .locator(`[data-testid="polls-table"] tbody tr td:nth-child(${columnIndex})`)
    .allTextContents();
}

function expectSortedStrings(values: string[], dir: "asc" | "desc") {
  const sorted = [...values].sort((a, b) => a.localeCompare(b));
  expect(values).toEqual(dir === "asc" ? sorted : sorted.reverse());
}

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

    // Click "Race" header to sort by race ascending
    await page.locator("th", { hasText: "Race" }).click({ force: true });

    await expect(page).toHaveURL(/\/polls\?sort=race&dir=asc$/);
    expectSortedStrings(await columnTexts(page, 2), "asc");
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
    await breadcrumb.locator("a", { hasText: "Home" }).click({ force: true });
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

  test("hydrates race ascending from URL sort params", async ({ page }) => {
    await gotoPolls(page, "?sort=race&dir=asc");

    await expect(page.locator("th", { hasText: "Race" })).toContainText("↑");
    expectSortedStrings(await columnTexts(page, 2), "asc");
  });

  test("hydrates grade descending from URL sort params", async ({ page }) => {
    await gotoPolls(page, "?sort=grade&dir=desc");

    await expect(page.locator("th", { hasText: "Grade" })).toContainText("↓");
    const grades = await columnTexts(page, 6);
    const sorted = [...grades].sort((a, b) => GRADE_ORDER[b] - GRADE_ORDER[a]);
    expect(grades).toEqual(sorted);
  });

  test("clicking a sort header updates URL without full page reload", async ({ page }) => {
    await gotoPolls(page);
    let documentRequests = 0;
    page.on("request", (request) => {
      if (request.isNavigationRequest() && request.resourceType() === "document") {
        documentRequests += 1;
      }
    });

    await page.locator("th", { hasText: "Race" }).click({ force: true });

    await expect(page).toHaveURL(/\/polls\?sort=race&dir=asc$/);
    expect(documentRequests).toBe(0);
    expectSortedStrings(await columnTexts(page, 2), "asc");
  });

  test("default date descending sort omits sort params", async ({ page }) => {
    await gotoPolls(page, "?sort=race&dir=asc");

    await page.locator("th", { hasText: "Date" }).click({ force: true });

    await expect(page).toHaveURL(/\/polls$/);
    await expect(page.locator("th", { hasText: "Date" })).toContainText("↓");
  });

  test("invalid sort param falls back to default without crashing", async ({ page }) => {
    await gotoPolls(page, "?sort=unknown&dir=asc");

    await expect(page).toHaveURL(/\/polls$/);
    await expect(page.locator("th", { hasText: "Date" })).toContainText("↓");
  });
});
