import useSWR from "swr";
import { fetchSuperTypes } from "@/lib/api";
import type { SuperTypeSummary } from "@/lib/types";

/** Super-type summaries — refreshes every 60 minutes (very stable, changes only on retrain). */
export function useSuperTypes() {
  return useSWR<SuperTypeSummary[]>("super-types", fetchSuperTypes, {
    revalidateOnFocus: false,
    dedupingInterval: 3_600_000, // 60 min
  });
}
