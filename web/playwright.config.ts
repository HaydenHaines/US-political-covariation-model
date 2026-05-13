import { defineConfig } from "@playwright/test";

const port = process.env.PORT ?? "3001";

export default defineConfig({
  testDir: "./e2e",
  /* Run tests sequentially to avoid overloading the API */
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://localhost:${port}`,
    headless: true,
  },
  webServer: {
    command: `PORT=${port} pnpm run dev`,
    url: `http://localhost:${port}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
