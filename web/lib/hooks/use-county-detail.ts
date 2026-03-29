import useSWR from "swr";
import { fetchCountyDetail } from "@/lib/api";
import type { CountyDetail } from "@/lib/types";

/**
 * Detail for a single county by FIPS code.
 * Pass null to skip fetching (conditional fetching pattern).
 * Refreshes every 30 minutes (stable between retrains).
 */
export function useCountyDetail(fips: string | null) {
  return useSWR<CountyDetail>(
    fips != null ? `county-${fips}` : null,
    () => fetchCountyDetail(fips!),
    {
      revalidateOnFocus: false,
      dedupingInterval: 1_800_000, // 30 min
    },
  );
}
