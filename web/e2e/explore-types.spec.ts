import { expect, test, type Page } from "@playwright/test";

const TYPES = [
  {
    type_id: 101,
    super_type_id: 1,
    display_name: "Dense urban centers",
    n_counties: 12,
    mean_pred_dem_share: 0.72,
    median_hh_income: 82000,
    pct_bachelors_plus: 0.44,
    pct_white_nh: 0.38,
    log_pop_density: 7.1,
  },
  {
    type_id: 202,
    super_type_id: 2,
    display_name: "Small town manufacturing",
    n_counties: 18,
    mean_pred_dem_share: 0.43,
    median_hh_income: 58000,
    pct_bachelors_plus: 0.24,
    pct_white_nh: 0.79,
    log_pop_density: 3.8,
  },
];

const TYPE_SCATTER = TYPES.map((type) => ({
  type_id: type.type_id,
  super_type_id: type.super_type_id,
  display_name: type.display_name,
  n_counties: type.n_counties,
  demographics: {
    mean_dem_share: type.mean_pred_dem_share,
    pct_white_nh: type.pct_white_nh,
    pct_bachelors_plus: type.pct_bachelors_plus,
    median_hh_income: type.median_hh_income,
    log_pop_density: type.log_pop_density,
  },
  shift_profile: {},
}));

const SUPER_TYPES = [
  {
    super_type_id: 1,
    display_name: "Urban core",
    member_type_ids: [101],
    n_counties: 12,
  },
  {
    super_type_id: 2,
    display_name: "Industrial towns",
    member_type_ids: [202],
    n_counties: 18,
  },
];

const XT_IMPACT = {
  top_movers: [],
  races_with_xt: 0,
  report_date: "2026-05-17",
};

async function mockExploreTypesData(page: Page) {
  await page.route("**/api/v1/types", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(TYPES),
    });
  });

  await page.route("**/api/v1/types/scatter-data", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(TYPE_SCATTER),
    });
  });

  await page.route("**/api/v1/super-types", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(SUPER_TYPES),
    });
  });

  await page.route("**/api/forecast/xt-impact?limit=10", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(XT_IMPACT),
    });
  });
}

async function gotoExploreTypes(page: Page, query = "") {
  await mockExploreTypesData(page);
  await page.goto(`/explore/types${query}`);
  await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {});
}

test.describe("Explore types page (/explore/types)", () => {
  test("page loads with main content", async ({ page }) => {
    await gotoExploreTypes(page);
    await expect(page.locator("main#main-content")).toBeVisible({ timeout: 10_000 });
  });

  test("page has an electoral types heading", async ({ page }) => {
    await gotoExploreTypes(page);
    const h1 = page.locator("h1").first();
    await expect(h1).toBeVisible({ timeout: 10_000 });
    await expect(h1).toContainText("Types");
  });

  test("renders at least one super-type group without a query", async ({ page }) => {
    await gotoExploreTypes(page);
    await page.getByTestId("type-group").first().waitFor({
      state: "attached",
      timeout: 10_000,
    });
  });

  test("hydrates the search input from the q URL parameter", async ({ page }) => {
    await gotoExploreTypes(page, "?q=urban");
    await expect(page.getByTestId("type-search")).toHaveValue("urban");
  });

  test("writes search input changes to the URL", async ({ page }) => {
    await gotoExploreTypes(page);
    await page.getByTestId("type-search").fill("urban");
    await expect(page).toHaveURL(/[?&]q=/);
  });

  test("clears the search input and removes q from the URL", async ({ page }) => {
    await gotoExploreTypes(page, "?q=urban");
    await page.getByTestId("type-search-clear").click();
    await expect(page).not.toHaveURL(/[?&]q=/);
    await expect(page.getByTestId("type-search")).toHaveValue("");
  });

  test("renders a no-results state for unmatched queries", async ({ page }) => {
    await gotoExploreTypes(page, "?q=not-a-real-type");
    await expect(page.getByTestId("type-empty-state")).toBeVisible();
  });

  test("does not emit console errors", async ({ page }) => {
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => pageErrors.push(err.message));

    await gotoExploreTypes(page);

    const realErrors = consoleErrors.filter(
      (e) =>
        !e.includes("net::ERR_") &&
        !e.includes("Failed to load resource") &&
        !e.includes("favicon"),
    );
    expect(realErrors).toHaveLength(0);
    expect(pageErrors).toHaveLength(0);
  });
});
