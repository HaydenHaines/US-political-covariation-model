import useSWR from "swr";
import { fetchPolls } from "@/lib/api";
import type { PollRow } from "@/lib/types";

export interface PollParams {
  race?: string;
  state?: string;
  cycle?: string;
}

/**
 * Polls filtered by optional race, state, and cycle.
 * Cache key includes all params to prevent stale data across different filter combinations.
 * Refreshes every 15 minutes (changes as polls are ingested).
 */
export function usePolls(params: PollParams = {}) {
  const { race, state, cycle } = params;
  // Build a stable, order-independent cache key from params
  const key = `polls|race=${race ?? ""}|state=${state ?? ""}|cycle=${cycle ?? ""}`;

  return useSWR<PollRow[]>(
    key,
    () => fetchPolls(params),
    {
      revalidateOnFocus: false,
      dedupingInterval: 900_000, // 15 min
    },
  );
}
