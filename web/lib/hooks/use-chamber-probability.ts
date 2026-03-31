import useSWR from "swr";
import { fetchChamberProbability, type ChamberProbabilityData } from "@/lib/api";

/**
 * Chamber control probability — computed via Monte Carlo on the server.
 * Refreshes every 10 minutes (the number only changes when predictions change,
 * which happens at most daily after a poll scrape run).
 */
export function useChamberProbability() {
  return useSWR<ChamberProbabilityData>(
    "chamber-probability",
    fetchChamberProbability,
    {
      revalidateOnFocus: false,
      dedupingInterval: 600_000, // 10 min
    },
  );
}
