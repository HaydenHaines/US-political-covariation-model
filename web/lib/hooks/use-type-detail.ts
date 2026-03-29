import useSWR from "swr";
import { fetchTypeDetail, type TypeDetail } from "@/lib/api";

/**
 * Detail for a single electoral type.
 * Pass null to skip fetching (conditional fetching pattern).
 * Refreshes every 30 minutes (stable between retrains).
 */
export function useTypeDetail(id: number | null) {
  return useSWR<TypeDetail>(
    id != null ? `type-${id}` : null,
    () => fetchTypeDetail(id!),
    {
      revalidateOnFocus: false,
      dedupingInterval: 1_800_000, // 30 min
    },
  );
}
