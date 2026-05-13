import { expect, type Page, test } from "@playwright/test";

type GovernorRaceGroupSnapshot = {
  key: string;
  states: string[];
  openSeatStates: string[];
  hrefs: string[];
};

const stateFilter = (page: Page) => page.getByTestId("state-filter");
const openSeatFilter = (page: Page) => page.getByTestId("open-seat-filter");
const governorRaceGroups = (page: Page) =>
  page.getByTestId("governor-race-group");

async function waitForGovernorPage(page: Page) {
  await page.goto("/forecast/governor");
  await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({
    timeout: 30_000,
  });
  await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });
}

async function groupSnapshots(page: Page): Promise<GovernorRaceGroupSnapshot[]> {
  const groups = governorRaceGroups(page);
  const count = await groups.count();
  const snapshots: GovernorRaceGroupSnapshot[] = [];

  for (let index = 0; index < count; index += 1) {
    const group = groups.nth(index);
    const key = await group.getAttribute("data-group");
    const states = (await group.getAttribute("data-states"))
      ?.split(" ")
      .filter(Boolean);
    const openSeatStates = (await group.getAttribute("data-open-seat-states"))
      ?.split(" ")
      .filter(Boolean);
    const links = group.locator('a[href*="-governor"]:visible');
    const hrefs = await links.evaluateAll((elements) =>
      elements
        .map((element) => element.getAttribute("href"))
        .filter((href): href is string => Boolean(href)),
    );

    expect(key).toBeTruthy();
    expect(states).toBeTruthy();
    expect(openSeatStates).toBeTruthy();
    expect(hrefs).toHaveLength(states?.length ?? 0);

    snapshots.push({
      key: key!,
      states: states!,
      openSeatStates: openSeatStates!,
      hrefs,
    });
  }

  return snapshots;
}

function chooseFilteringNeedle(groups: GovernorRaceGroupSnapshot[]) {
  const total = groups.reduce((sum, group) => sum + group.states.length, 0);
  const candidates = ["a", "n", "o", "e", "i", "m", "c"];

  for (const candidate of candidates) {
    const filteredTotal = groups.reduce(
      (sum, group) =>
        sum +
        group.states.filter((state) =>
          state.toLowerCase().includes(candidate),
        ).length,
      0,
    );

    if (filteredTotal > 0 && filteredTotal < total) {
      return candidate;
    }
  }

  throw new Error("Could not find a state substring that filters governor races");
}

function filterSnapshotGroups(
  groups: GovernorRaceGroupSnapshot[],
  {
    stateNeedle = "",
    openSeatsOnly = false,
  }: { stateNeedle?: string; openSeatsOnly?: boolean },
) {
  const normalizedNeedle = stateNeedle.toLowerCase();

  return groups
    .map((group) => {
      const openSeatSet = new Set(group.openSeatStates);
      const states = group.states.filter((state) => {
        const matchesState =
          !normalizedNeedle || state.toLowerCase().includes(normalizedNeedle);
        const matchesOpenSeat = !openSeatsOnly || openSeatSet.has(state);

        return matchesState && matchesOpenSeat;
      });

      return {
        ...group,
        states,
        openSeatStates: group.openSeatStates.filter((state) =>
          states.includes(state),
        ),
        hrefs: group.hrefs.filter((href) =>
          states.some((state) => href.includes(`-${state.toLowerCase()}-`)),
        ),
      };
    })
    .filter((group) => group.states.length > 0);
}

function chooseClosedSeatState(groups: GovernorRaceGroupSnapshot[]) {
  for (const group of groups) {
    const openSeatSet = new Set(group.openSeatStates);
    const closedSeatState = group.states.find((state) => !openSeatSet.has(state));

    if (closedSeatState) {
      return closedSeatState;
    }
  }

  throw new Error("Could not find a non-open governor race");
}

test.describe("Governor overview page", () => {
  test.describe("URL sync", () => {
    test("SC1: ?state=a hydrates state input and narrows race groups on direct load", async ({ page }) => {
      await page.goto("/forecast/governor?state=a");
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });
      await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });

      await expect(stateFilter(page)).toHaveValue("a");
      await expect(page.getByTestId("governor-filter-empty-state")).not.toBeVisible();

      const groups = await groupSnapshots(page);
      expect(groups.length).toBeGreaterThan(0);
      for (const group of groups) {
        expect(group.states.every((s) => s.toLowerCase().includes("a"))).toBe(true);
      }
    });

    test("SC2: ?openSeat=1 hydrates checkbox and groups show only open-seat states", async ({ page }) => {
      await page.goto("/forecast/governor?openSeat=1");
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });
      await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });

      await expect(openSeatFilter(page)).toBeChecked();

      const groups = await groupSnapshots(page);
      expect(groups.length).toBeGreaterThan(0);
      for (const group of groups) {
        expect(group.states).toEqual(group.openSeatStates);
      }
    });

    test("SC3: ?state and ?openSeat=1 compose correctly on direct load", async ({ page }) => {
      await waitForGovernorPage(page);
      const initialGroups = await groupSnapshots(page);
      const openSeatState = initialGroups.flatMap((g) => g.openSeatStates).find(Boolean);
      expect(openSeatState).toBeTruthy();

      const needle = openSeatState!.toLowerCase();
      await page.goto(`/forecast/governor?state=${encodeURIComponent(needle)}&openSeat=1`);
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });
      await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });

      await expect(stateFilter(page)).toHaveValue(needle);
      await expect(openSeatFilter(page)).toBeChecked();

      const filteredGroups = await groupSnapshots(page);
      const expectedGroups = filterSnapshotGroups(initialGroups, {
        stateNeedle: openSeatState!,
        openSeatsOnly: true,
      });
      expect(filteredGroups.map((g) => g.key)).toEqual(expectedGroups.map((g) => g.key));
    });

    test("SC4: typing state filter updates ?state param (debounced) and preserves unrelated params", async ({ page }) => {
      await page.goto("/forecast/governor?view=test");
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });
      await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });

      await stateFilter(page).fill("a");

      await expect
        .poll(() => new URL(page.url()).searchParams.get("state"), { timeout: 1_000 })
        .toBe("a");
      expect(new URL(page.url()).searchParams.get("view")).toBe("test");

      await stateFilter(page).fill("");
      await expect
        .poll(() => new URL(page.url()).searchParams.get("state"), { timeout: 1_000 })
        .toBeNull();
      expect(new URL(page.url()).searchParams.get("view")).toBe("test");
    });

    test("SC5: checkbox toggles ?openSeat=1 immediately and preserves unrelated params", async ({ page }) => {
      await page.goto("/forecast/governor?view=test&state=a");
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });
      await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });

      await openSeatFilter(page).check();
      await expect
        .poll(() => new URL(page.url()).searchParams.get("openSeat"), { timeout: 500 })
        .toBe("1");
      expect(new URL(page.url()).searchParams.get("view")).toBe("test");
      expect(new URL(page.url()).searchParams.get("state")).toBe("a");

      await openSeatFilter(page).uncheck();
      await expect
        .poll(() => new URL(page.url()).searchParams.get("openSeat"), { timeout: 500 })
        .toBeNull();
      expect(new URL(page.url()).searchParams.get("view")).toBe("test");
      expect(new URL(page.url()).searchParams.get("state")).toBe("a");
    });

    test("SC6: browser back restores URL, state input, and rendered groups", async ({ page }) => {
      await page.goto("/forecast/governor?state=a");
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });
      await expect(governorRaceGroups(page).first()).toBeVisible({ timeout: 15_000 });
      await expect(stateFilter(page)).toHaveValue("a");

      await stateFilter(page).fill("n");
      await expect
        .poll(() => new URL(page.url()).searchParams.get("state"), { timeout: 1_000 })
        .toBe("n");

      await page.goBack();

      await expect
        .poll(() => new URL(page.url()).searchParams.get("state"), { timeout: 1_000 })
        .toBe("a");
      await expect(stateFilter(page)).toHaveValue("a");
    });

    test("SC7: unmatched direct load renders empty state while FundamentalsCard stays mounted", async ({ page }) => {
      await page.goto("/forecast/governor?state=XYZZY_NO_MATCH&openSeat=1");
      await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible({ timeout: 30_000 });

      await expect(page.getByTestId("governor-filter-empty-state")).toBeVisible({ timeout: 10_000 });
      await expect(governorRaceGroups(page)).toHaveCount(0);
      await expect(page.locator('[aria-label="National Environment"]')).toBeVisible();
    });
  });

  test("governor page loads and shows heading", async ({ page }) => {
    await waitForGovernorPage(page);
    await expect(page.getByRole("heading", { name: /Governor Races/ })).toBeVisible();
  });

  test("governor page shows race cards", async ({ page }) => {
    await waitForGovernorPage(page);

    const snapshots = await groupSnapshots(page);
    const raceCardCount = snapshots.reduce(
      (sum, group) => sum + group.hrefs.length,
      0,
    );
    expect(raceCardCount).toBeGreaterThan(0);
  });

  test("race card links point to governor detail pages", async ({ page }) => {
    await waitForGovernorPage(page);

    const firstLink = governorRaceGroups(page)
      .first()
      .locator('a[href*="-governor"]:visible')
      .first();
    await expect(firstLink).toBeVisible();
    const href = await firstLink.getAttribute("href");
    expect(href).toMatch(/\/forecast\/2026-[a-z]+-governor/);
  });

  // SC1: state-filter input is visible between header <p> and FundamentalsCard
  test("state filter and open-seat filter are visible on page load", async ({ page }) => {
    await waitForGovernorPage(page);
    await expect(stateFilter(page)).toBeVisible();
    await expect(openSeatFilter(page)).toBeVisible();
  });

  // SC2: lowercase substring filtering is applied consistently to every RaceCardGrid
  test("typing a lowercase state substring filters every race group", async ({
    page,
  }) => {
    await waitForGovernorPage(page);
    const initialGroups = await groupSnapshots(page);
    const needle = chooseFilteringNeedle(initialGroups);

    await stateFilter(page).fill(needle);

    const filteredGroups = await groupSnapshots(page);
    const expectedGroups = filterSnapshotGroups(initialGroups, {
      stateNeedle: needle,
    });

    expect(filteredGroups.map((group) => group.key)).toEqual(
      expectedGroups.map((group) => group.key),
    );

    for (const filteredGroup of filteredGroups) {
      expect(filteredGroup.states.length).toBeGreaterThan(0);
      expect(
        filteredGroup.states.every((state) =>
          state.toLowerCase().includes(needle),
        ),
      ).toBe(true);
      expect(filteredGroup.hrefs).toHaveLength(filteredGroup.states.length);
    }
  });

  test("open-seat toggle filters every race group to open seats", async ({
    page,
  }) => {
    await waitForGovernorPage(page);
    const initialGroups = await groupSnapshots(page);

    await openSeatFilter(page).check();

    const filteredGroups = await groupSnapshots(page);
    const expectedGroups = filterSnapshotGroups(initialGroups, {
      openSeatsOnly: true,
    });

    expect(filteredGroups.map((group) => group.key)).toEqual(
      expectedGroups.map((group) => group.key),
    );

    for (const filteredGroup of filteredGroups) {
      expect(filteredGroup.states.length).toBeGreaterThan(0);
      expect(filteredGroup.states).toEqual(filteredGroup.openSeatStates);
      expect(filteredGroup.hrefs).toHaveLength(filteredGroup.states.length);
    }
  });

  test("open-seat toggle composes with the state substring filter", async ({
    page,
  }) => {
    await waitForGovernorPage(page);
    const initialGroups = await groupSnapshots(page);
    const openSeatState = initialGroups
      .flatMap((group) => group.openSeatStates)
      .find(Boolean);

    expect(openSeatState).toBeTruthy();

    await openSeatFilter(page).check();
    await stateFilter(page).fill(openSeatState!.toLowerCase());

    const filteredGroups = await groupSnapshots(page);
    const expectedGroups = filterSnapshotGroups(initialGroups, {
      stateNeedle: openSeatState!,
      openSeatsOnly: true,
    });

    expect(filteredGroups).toEqual(expectedGroups);
  });

  // SC3: clearing filters restores all groups and preserves RaceCardGrid ordering
  test("clearing filters restores all race groups in their original order", async ({
    page,
  }) => {
    await waitForGovernorPage(page);
    const initialGroups = await groupSnapshots(page);
    const needle = chooseFilteringNeedle(initialGroups);

    await openSeatFilter(page).check();
    await stateFilter(page).fill(needle);
    await expect(governorRaceGroups(page).first()).toBeVisible();

    await openSeatFilter(page).uncheck();
    await stateFilter(page).fill("");
    await expect(governorRaceGroups(page)).toHaveCount(initialGroups.length);
    await expect(page.getByTestId("governor-filter-empty-state")).toHaveCount(0);

    const restoredGroups = await groupSnapshots(page);
    expect(restoredGroups).toEqual(initialGroups);
  });

  // SC4: no-match input shows empty-state message; FundamentalsCard stays mounted
  test("no-match filter shows empty state and keeps FundamentalsCard", async ({
    page,
  }) => {
    await waitForGovernorPage(page);

    const initialGroups = await groupSnapshots(page);
    const closedSeatState = chooseClosedSeatState(initialGroups);

    await openSeatFilter(page).check();
    await stateFilter(page).fill(closedSeatState.toLowerCase());

    await expect(governorRaceGroups(page)).toHaveCount(0);
    await expect(page.getByTestId("governor-filter-empty-state")).toContainText(
      "No governor races match the selected filters.",
    );
    await expect(page.locator('[aria-label="National Environment"]')).toBeVisible();
  });
});
