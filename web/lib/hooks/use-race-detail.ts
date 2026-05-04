import useSWR from "swr";
import { fetchRaceDetail, type RaceDetail } from "@/lib/api";

/** Fetch race detail data for a single race. Refreshes every 5 minutes. */
export function useRaceDetail(slug: string) {
  return useSWR<RaceDetail>(
    slug ? `race-detail|${slug}` : null,
    () => fetchRaceDetail(slug),
    {
      revalidateOnFocus: false,
      dedupingInterval: 300_000, // 5 min
    },
  );
}
