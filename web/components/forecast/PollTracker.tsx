"use client";

import { usePolls } from "@/lib/hooks/use-polls";
import { formatMargin } from "@/lib/format";
import { Skeleton } from "@/components/ui/skeleton";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface PollTrackerProps {
  /** State abbreviation to filter polls (e.g. "GA"). */
  stateAbbr: string;
  /** Race label to filter polls (e.g. "2026 GA Senate"). */
  race: string;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  try {
    return new Date(dateStr + "T12:00:00").toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return dateStr;
  }
}

/**
 * Poll tracker — fetches polls via SWR and renders a simple list.
 * Intended for embedding in the race detail page.
 */
export function PollTracker({ stateAbbr, race }: PollTrackerProps) {
  const { data: polls, isLoading, error } = usePolls({
    state: stateAbbr,
    race,
  });

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[0, 1, 2].map((i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-sm text-muted-foreground">
        Unable to load polls.
      </p>
    );
  }

  if (!polls || polls.length === 0) {
    return (
      <p className="text-sm text-muted-foreground italic">
        No polls available yet for this race.
      </p>
    );
  }

  // Sort newest first
  const sorted = [...polls].sort((a, b) => {
    if (!a.date) return 1;
    if (!b.date) return -1;
    return b.date.localeCompare(a.date);
  });

  return (
    <div>
      {/* Mobile: trend line — date + margin + direction indicator */}
      <div className="md:hidden space-y-1">
        {sorted.map((poll, i) => {
          const margin = formatMargin(poll.dem_share);
          const isDem = poll.dem_share > 0.505;
          const isGop = poll.dem_share < 0.495;
          const color = isDem
            ? "var(--forecast-safe-d)"
            : isGop
            ? "var(--forecast-safe-r)"
            : "var(--forecast-tossup)";
          const TrendIcon = isDem ? TrendingUp : isGop ? TrendingDown : Minus;

          return (
            <div
              key={i}
              className="flex items-center justify-between py-2 text-sm"
              style={{ borderBottom: "1px solid var(--color-border)" }}
            >
              <span style={{ color: "var(--color-text-muted)", minWidth: 80 }}>
                {formatDate(poll.date)}
              </span>
              <TrendIcon size={14} style={{ color, flexShrink: 0 }} aria-hidden />
              <span
                className="font-mono font-semibold ml-auto"
                style={{ color }}
              >
                {margin}
              </span>
            </div>
          );
        })}
      </div>

      {/* Desktop: full table (≥768px) */}
      <div className="hidden md:block overflow-x-auto">
        <table
          className="w-full text-sm border-collapse"
          aria-label="Recent polls"
        >
          <thead>
            <tr
              className="border-b text-muted-foreground text-xs font-semibold uppercase tracking-wide"
              style={{ borderColor: "var(--color-border)" }}
            >
              <th className="text-left py-2 pr-3">Date</th>
              <th className="text-left py-2 px-3">Pollster</th>
              <th className="text-right py-2 px-3">Margin</th>
              <th className="text-right py-2 pl-3">Sample</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((poll, i) => {
              const margin = formatMargin(poll.dem_share);
              const isDem = poll.dem_share > 0.505;
              const isGop = poll.dem_share < 0.495;
              const color = isDem
                ? "var(--forecast-safe-d)"
                : isGop
                ? "var(--forecast-safe-r)"
                : "var(--forecast-tossup)";
              return (
                <tr
                  key={i}
                  style={{
                    borderBottom: "1px solid var(--color-bg)",
                  }}
                >
                  <td
                    className="py-2 pr-3 text-muted-foreground"
                    style={{ color: "var(--color-text-muted)" }}
                  >
                    {formatDate(poll.date)}
                  </td>
                  <td className="py-2 px-3">{poll.pollster ?? "Unknown"}</td>
                  <td
                    className="text-right py-2 px-3 font-mono font-semibold"
                    style={{ color }}
                  >
                    {margin}
                  </td>
                  <td
                    className="text-right py-2 pl-3"
                    style={{ color: "var(--color-text-muted)" }}
                  >
                    {poll.n_sample !== null
                      ? poll.n_sample.toLocaleString("en-US")
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
