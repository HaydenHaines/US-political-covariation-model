import useSWR from "swr";
import { fetchGovernorOverview, type GovernorOverviewData } from "@/lib/api";

/** Governor overview data — refreshes every 5 minutes (changes when polls arrive). */
export function useGovernorOverview() {
  return useSWR<GovernorOverviewData>("governor-overview", fetchGovernorOverview, {
    revalidateOnFocus: false,
    dedupingInterval: 300_000, // 5 min
  });
}
