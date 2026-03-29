import useSWR from "swr";
import { fetchSenateOverview, type SenateOverviewData } from "@/lib/api";

/** Senate overview data — refreshes every 5 minutes (changes when polls arrive). */
export function useSenateOverview() {
  return useSWR<SenateOverviewData>("senate-overview", fetchSenateOverview, {
    revalidateOnFocus: false,
    dedupingInterval: 300_000, // 5 min
  });
}
