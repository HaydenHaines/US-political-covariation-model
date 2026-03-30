import useSWR from "swr";
import { fetchPollTrend } from "@/lib/api";
import type { PollTrendResponse } from "@/lib/types";

/**
 * Fetch poll trend data for a race detail page.
 * Keyed by slug; refreshes every 15 minutes.
 */
export function usePollTrend(slug: string) {
  return useSWR<PollTrendResponse>(
    slug ? `poll-trend|${slug}` : null,
    () => fetchPollTrend(slug),
    {
      revalidateOnFocus: false,
      dedupingInterval: 900_000, // 15 min
    },
  );
}
