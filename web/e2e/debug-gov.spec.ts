import { test } from "@playwright/test";

test("governor network debug", async ({ page }) => {
  const failures: string[] = [];
  
  page.on("response", (resp) => {
    if (resp.status() >= 400) {
      failures.push(`${resp.status()} ${resp.url()}`);
    }
  });

  await page.goto("/forecast/governor");
  await page.waitForTimeout(8000);
  
  console.log("Failed requests (400+):", JSON.stringify(failures, null, 2));
});
