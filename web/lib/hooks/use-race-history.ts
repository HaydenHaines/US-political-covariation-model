import useSWR from "swr";
import { fetchRaceHistory, type RaceHistoryEntry } from "@/lib/api";

/**
 * Fetch per-race margin history for sparkline rendering.
 *
 * Returns one entry per race, each with a chronological array of
 * {date, margin} points sourced from forecast snapshots.
 *
 * Refreshes every 15 minutes -- snapshots are added twice-weekly by cron.
 * Index by slug for O(1) lookup from race cards.
 */
export function useRaceHistory() {
  const swr = useSWR<RaceHistoryEntry[]>("race-history", fetchRaceHistory, {
    revalidateOnFocus: false,
    dedupingInterval: 900_000, // 15 min
  });

  // Build slug to history lookup so consumers don't need to search the array.
  const historyBySlug = new Map<string, RaceHistoryEntry["history"]>(
    (swr.data ?? []).map((entry) => [entry.slug, entry.history]),
  );

  return { ...swr, historyBySlug };
}
