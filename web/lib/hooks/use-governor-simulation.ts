import useSWR from "swr";
import { fetchGovernorSimulation, type GovernorSimulationData } from "@/lib/api";

/** Governor Monte Carlo simulation — seat distribution across all 36 races. Refreshes every 5 minutes. */
export function useGovernorSimulation() {
  return useSWR<GovernorSimulationData>(
    "governor-simulation",
    fetchGovernorSimulation,
    {
      revalidateOnFocus: false,
      dedupingInterval: 300_000, // 5 min
    },
  );
}
