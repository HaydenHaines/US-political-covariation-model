import useSWR from "swr";
import { fetchTypes } from "@/lib/api";
import type { TypeSummary } from "@/lib/types";

/** All electoral type summaries — refreshes every 30 minutes (changes only on retrain). */
export function useTypes() {
  return useSWR<TypeSummary[]>("types", fetchTypes, {
    revalidateOnFocus: false,
    dedupingInterval: 1_800_000, // 30 min
  });
}
