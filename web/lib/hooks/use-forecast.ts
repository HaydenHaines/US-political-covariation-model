import useSWR from "swr";
import { fetchForecast, type ForecastRow } from "@/lib/api";

/**
 * Forecast data filtered by optional race and state.
 * Cache key includes both params so different filter combinations don't collide.
 * Refreshes every 5 minutes (changes as polls are ingested).
 */
export function useForecast(race?: string, state?: string) {
  const key = `forecast|race=${race ?? ""}|state=${state ?? ""}`;

  return useSWR<ForecastRow[]>(
    key,
    () => fetchForecast(race, state),
    {
      revalidateOnFocus: false,
      dedupingInterval: 300_000, // 5 min
    },
  );
}
