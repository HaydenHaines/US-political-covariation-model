import { expect, test, type Page } from "@playwright/test";

const TYPE_SCATTER = [
  {
    type_id: 101,
    super_type_id: 1,
    display_name: "Dense city centers",
    n_counties: 12,
    demographics: {},
    shift_profile: {
      pres_d_shift_00_04: 0.02,
      pres_d_shift_04_08: 0.08,
      pres_d_shift_08_12: -0.01,
      pres_d_shift_12_16: -0.06,
      pres_d_shift_16_20: 0.04,
      pres_d_shift_20_24: -0.02,
    },
  },
  {
    type_id: 202,
    super_type_id: 2,
    display_name: "Small town manufacturing",
    n_counties: 18,
    demographics: {},
    shift_profile: {
      pres_d_shift_00_04: -0.01,
      pres_d_shift_04_08: 0.03,
      pres_d_shift_08_12: -0.04,
      pres_d_shift_12_16: -0.09,
      pres_d_shift_16_20: 0.01,
      pres_d_shift_20_24: -0.03,
    },
  },
];

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

async function mockShiftData(page: Page) {
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
}

test.describe("Historical shifts page (/explore/shifts)", () => {
  test("renders the page shell and small multiples without console errors", async ({ page }) => {
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => pageErrors.push(err.message));

    await mockShiftData(page);
    await page.goto("/explore/shifts");

    await expect(page).toHaveURL(/\/explore\/shifts$/);
    await expect(page).toHaveTitle(/Historical Shifts by Super-Type/);
    await expect(page.locator("main#main-content")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole("heading", { name: "Historical Shifts" })).toBeVisible();
    await expect(page.getByText("Dem shift (positive = Dem gain)")).toBeVisible();

    const charts = page.locator('svg[role="img"]');
    await expect(charts).toHaveCount(2, { timeout: 10_000 });
    await expect(
      page.getByRole("img", {
        name: "Urban core presidential Dem shift across election cycles",
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("img", {
        name: "Industrial towns presidential Dem shift across election cycles",
      }),
    ).toBeVisible();
    await expect(page.locator("circle")).toHaveCount(12);

    expect(consoleErrors.filter((e) => !e.includes("favicon"))).toHaveLength(0);
    expect(pageErrors).toHaveLength(0);
  });

  test("breadcrumb links back to the explore surface", async ({ page }) => {
    await mockShiftData(page);
    await page.goto("/explore/shifts");

    const breadcrumb = page.getByRole("navigation", { name: "Breadcrumb" });
    await expect(breadcrumb).toBeVisible({ timeout: 10_000 });
    const exploreLink = breadcrumb.getByRole("link", { name: "Explore", exact: true });
    await expect(exploreLink).toHaveAttribute("href", "/explore");

    await exploreLink.click();
    await expect(page).toHaveURL(/\/explore$/);
  });
});
