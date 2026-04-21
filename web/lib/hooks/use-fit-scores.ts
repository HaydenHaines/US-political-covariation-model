import useSWR from "swr";
import { fetchFitScores, type FitScoreResponse } from "@/lib/api";

/**
 * Fetch Moneyball fit scores for a race/party combination.
 *
 * Fit scores are computed from the sabermetrics pipeline (career CTOV dot
 * product against district W vector) and only change when the model is
 * retrained.  A long deduping interval is intentional.
 *
 * @param raceKey  Race key, e.g. "2026 GA Senate"
 * @param party    "D" or "R" — filters results to one party's candidates
 * @param minRaces Minimum races required for a candidate to be included (default 2)
 */
export function useFitScores(
  raceKey: string | null,
  party: "D" | "R" | null,
  minRaces = 2,
) {
  return useSWR<FitScoreResponse>(
    raceKey && party ? ["fit-scores", raceKey, party, minRaces] : null,
    () => fetchFitScores(raceKey!, party!, minRaces),
    {
      revalidateOnFocus: false,
      // Fit scores change only on pipeline re-runs — 1 hour dedup is safe
      dedupingInterval: 3_600_000,
    },
  );
}
