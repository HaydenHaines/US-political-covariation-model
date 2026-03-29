import useSWR from "swr";
import { fetchTypeScatterData } from "@/lib/api";
import type { TypeScatterPoint } from "@/lib/types";

/** Type scatter data for the types explorer — refreshes every 30 minutes (stable between retrains). */
export function useTypeScatter() {
  return useSWR<TypeScatterPoint[]>("type-scatter", fetchTypeScatterData, {
    revalidateOnFocus: false,
    dedupingInterval: 1_800_000, // 30 min
  });
}
