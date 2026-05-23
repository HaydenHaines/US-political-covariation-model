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
  {
    type_id: 303,
    super_type_id: 3,
    display_name: "Suburban college towns",
    n_counties: 22,
    mean_pred_dem_share: 0.56,
    median_hh_income: 71000,
    pct_bachelors_plus: 0.52,
    pct_white_nh: 0.62,
    log_pop_density: 5.2,
  },
  {
    type_id: 404,
    super_type_id: 4,
    display_name: "Rural agricultural",
    n_counties: 45,
    mean_pred_dem_share: 0.32,
    median_hh_income: 49000,
    pct_bachelors_plus: 0.18,
    pct_white_nh: 0.88,
    log_pop_density: 2.1,
  },
  {
    type_id: 505,
    super_type_id: 5,
    display_name: "Exurban mixed",
    n_counties: 31,
    mean_pred_dem_share: 0.47,
    median_hh_income: 65000,
    pct_bachelors_plus: 0.28,
    pct_white_nh: 0.71,
    log_pop_density: 4.4,
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
  {
    super_type_id: 3,
    display_name: "College towns",
    member_type_ids: [303],
    n_counties: 22,
  },
  {
    super_type_id: 4,
    display_name: "Agricultural",
    member_type_ids: [404],
    n_counties: 45,
  },
  {
    super_type_id: 5,
    display_name: "Exurban",
    member_type_ids: [505],
    n_counties: 31,
  },
];

// TypeDetail mock data for /api/v1/types/:id — used by ComparisonTable
const TYPE_DETAILS: Record<number, Record<string, unknown>> = {
  101: {
    type_id: 101, super_type_id: 1, display_name: "Dense urban centers",
    n_counties: 12, mean_pred_dem_share: 0.72, median_hh_income: 82000,
    pct_bachelors_plus: 0.44, pct_white_nh: 0.38, log_pop_density: 7.1,
    counties: [],
    demographics: { mean_dem_share: 0.72, pct_white_nh: 0.38, pct_bachelors_plus: 0.44, median_hh_income: 82000 },
    shift_profile: {},
    narrative: null,
  },
  202: {
    type_id: 202, super_type_id: 2, display_name: "Small town manufacturing",
    n_counties: 18, mean_pred_dem_share: 0.43, median_hh_income: 58000,
    pct_bachelors_plus: 0.24, pct_white_nh: 0.79, log_pop_density: 3.8,
    counties: [],
    demographics: { mean_dem_share: 0.43, pct_white_nh: 0.79, pct_bachelors_plus: 0.24, median_hh_income: 58000 },
    shift_profile: {},
    narrative: null,
  },
  303: {
    type_id: 303, super_type_id: 3, display_name: "Suburban college towns",
    n_counties: 22, mean_pred_dem_share: 0.56, median_hh_income: 71000,
    pct_bachelors_plus: 0.52, pct_white_nh: 0.62, log_pop_density: 5.2,
    counties: [],
    demographics: { mean_dem_share: 0.56, pct_white_nh: 0.62, pct_bachelors_plus: 0.52, median_hh_income: 71000 },
    shift_profile: {},
    narrative: null,
  },
  404: {
    type_id: 404, super_type_id: 4, display_name: "Rural agricultural",
    n_counties: 45, mean_pred_dem_share: 0.32, median_hh_income: 49000,
    pct_bachelors_plus: 0.18, pct_white_nh: 0.88, log_pop_density: 2.1,
    counties: [],
    demographics: { mean_dem_share: 0.32, pct_white_nh: 0.88, pct_bachelors_plus: 0.18, median_hh_income: 49000 },
    shift_profile: {},
    narrative: null,
  },
  505: {
    type_id: 505, super_type_id: 5, display_name: "Exurban mixed",
    n_counties: 31, mean_pred_dem_share: 0.47, median_hh_income: 65000,
    pct_bachelors_plus: 0.28, pct_white_nh: 0.71, log_pop_density: 4.4,
    counties: [],
    demographics: { mean_dem_share: 0.47, pct_white_nh: 0.71, pct_bachelors_plus: 0.28, median_hh_income: 65000 },
    shift_profile: {},
    narrative: null,
  },
};

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

  // Individual type detail routes — used by ComparisonTable's useTypeDetail hook
  for (const id of [101, 202, 303, 404, 505]) {
    const detail = TYPE_DETAILS[id];
    await page.route(`**/api/v1/types/${id}`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(detail),
      });
    });
  }
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

test.describe("ScatterPlot axis URL sync (/explore/types v23)", () => {
  test("scatter axis selects render with default axes (pct_white_nh / mean_dem_share)", async ({
    page,
  }) => {
    await gotoExploreTypes(page);
    await expect(page.getByTestId("scatter-x-select")).toHaveValue("pct_white_nh");
    await expect(page.getByTestId("scatter-y-select")).toHaveValue("mean_dem_share");
  });

  test("hydrates x and y axis selects from URL parameters", async ({ page }) => {
    await gotoExploreTypes(page, "?x=pct_black&y=median_age");
    await expect(page.getByTestId("scatter-x-select")).toHaveValue("pct_black");
    await expect(page.getByTestId("scatter-y-select")).toHaveValue("median_age");
  });

  test("writes axis selection change back to the URL", async ({ page }) => {
    await gotoExploreTypes(page);
    await page.getByTestId("scatter-x-select").selectOption("pct_black");
    await expect(page).toHaveURL(/[?&]x=pct_black/);
  });

  test("omits default axis value from URL on round-trip back to default", async ({ page }) => {
    await gotoExploreTypes(page);
    // Change to non-default so x lands in the URL
    await page.getByTestId("scatter-x-select").selectOption("pct_black");
    await expect(page).toHaveURL(/[?&]x=pct_black/);
    // Change back to the default — x should be removed from the URL
    await page.getByTestId("scatter-x-select").selectOption("pct_white_nh");
    await expect(page).not.toHaveURL(/[?&]x=/);
  });

  test("falls back to default x axis for an invalid URL parameter value", async ({ page }) => {
    await gotoExploreTypes(page, "?x=not_a_real_field");
    await expect(page.getByTestId("scatter-x-select")).toHaveValue("pct_white_nh");
  });
});

test.describe("ComparisonTable URL sync (v24)", () => {
  test("hydrates selected type columns from ?types=101,202", async ({ page }) => {
    await gotoExploreTypes(page, "?types=101,202");
    const table = page.getByTestId("comparison-table");
    await expect(table.getByText("Dense urban centers")).toBeVisible({ timeout: 10_000 });
    await expect(table.getByText("Small town manufacturing")).toBeVisible({ timeout: 10_000 });
  });

  test("selecting a type from combobox writes types= to URL", async ({ page }) => {
    await gotoExploreTypes(page);
    const selector = page.getByTestId("comparison-type-selector");
    await expect(selector).toBeVisible({ timeout: 10_000 });
    await selector.click();
    await selector.fill("101");
    await page.getByRole("option", { name: /Dense urban centers/ }).click();
    await expect(page).toHaveURL(/[?&]types=101/, { timeout: 5_000 });
  });

  test("adding second type preserves ?q=, ?x=, ?y= state", async ({ page }) => {
    await gotoExploreTypes(page, "?q=urban&x=pct_black&y=median_age");
    const selector = page.getByTestId("comparison-type-selector");
    await expect(selector).toBeVisible({ timeout: 10_000 });
    await selector.click();
    await selector.fill("Dense");
    await page.getByRole("option", { name: /Dense urban centers/ }).click();
    await expect(page).toHaveURL(/[?&]types=101/, { timeout: 5_000 });
    await expect(page).toHaveURL(/[?&]q=urban/);
    await expect(page).toHaveURL(/[?&]x=pct_black/);
    await expect(page).toHaveURL(/[?&]y=median_age/);
  });

  test("removing one type updates types= without clearing unrelated params", async ({ page }) => {
    await gotoExploreTypes(page, "?types=101,202&q=urban");
    const table = page.getByTestId("comparison-table");
    await expect(table.getByText("Dense urban centers")).toBeVisible({ timeout: 10_000 });
    await page.getByRole("button", { name: "Remove Dense urban centers" }).click();
    await expect(page).toHaveURL(/[?&]q=urban/, { timeout: 5_000 });
    await expect(page).toHaveURL(/[?&]types=202/);
    await expect(page).not.toHaveURL(/types=101,202/);
  });

  test("Clear all removes only the types= parameter", async ({ page }) => {
    await gotoExploreTypes(page, "?types=101,202&q=test");
    const table = page.getByTestId("comparison-table");
    await expect(table.getByText("Dense urban centers")).toBeVisible({ timeout: 10_000 });
    await page.getByTestId("comparison-clear-all").click();
    await expect(page).not.toHaveURL(/[?&]types=/, { timeout: 5_000 });
    await expect(page).toHaveURL(/[?&]q=test/);
  });

  test("invalid and duplicate type IDs from URL do not crash the page", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("pageerror", (err) => consoleErrors.push(err.message));

    await gotoExploreTypes(page, "?types=abc,101,101,xyz,202");
    await expect(page.locator("main#main-content")).toBeVisible({ timeout: 10_000 });

    const table = page.getByTestId("comparison-table");
    // Only valid unique IDs: 101 and 202
    await expect(table.getByText("Dense urban centers")).toBeVisible({ timeout: 10_000 });
    await expect(table.getByText("Small town manufacturing")).toBeVisible({ timeout: 10_000 });
    // No page-level JS errors
    expect(consoleErrors).toHaveLength(0);
  });

  test("more than four IDs are capped at four", async ({ page }) => {
    await gotoExploreTypes(page, "?types=101,202,303,404,505");
    const table = page.getByTestId("comparison-table");
    // First 4 types should be shown
    await expect(table.getByText("Dense urban centers")).toBeVisible({ timeout: 10_000 });
    await expect(table.getByText("Rural agricultural")).toBeVisible({ timeout: 10_000 });
    // 5th type must be absent from the comparison table
    await expect(table.getByText("Exurban mixed")).not.toBeVisible();
    // useEffect normalises the URL to 4 IDs on mount — 505 must be gone
    await expect(page).not.toHaveURL(/505/, { timeout: 5_000 });
    // Exactly 4 remove buttons confirms 4 types are loaded (not 5)
    await expect(table.getByRole("button", { name: /^Remove/ })).toHaveCount(4);
  });
});
