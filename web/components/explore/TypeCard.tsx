"use client";

import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import type { TypeSummary } from "@/lib/types";

interface TypeCardProps {
  type: TypeSummary;
}

/** Strip trailing "Red" or "Blue" suffix from a type display name. */
function stripPartySuffix(name: string): string {
  return name.replace(/\s+(Red|Blue)$/, "");
}

function formatIncome(val: number | null): string {
  if (val === null) return null as unknown as string;
  return `$${Math.round(val / 1_000)}K`;
}

function formatPct(val: number | null): string {
  if (val === null) return null as unknown as string;
  return `${(val * 100).toFixed(0)}%`;
}

/** Individual type card — links to /type/[id]. */
export function TypeCard({ type }: TypeCardProps) {
  const displayName = stripPartySuffix(type.display_name);

  const stats: { label: string; value: string }[] = [];
  if (type.n_counties > 0) {
    stats.push({ label: "counties", value: String(type.n_counties) });
  }
  const income = formatIncome(type.median_hh_income);
  if (income) stats.push({ label: "income", value: income });
  const college = formatPct(type.pct_bachelors_plus);
  if (college) stats.push({ label: "college", value: college });
  const white = formatPct(type.pct_white_nh);
  if (white) stats.push({ label: "white NH", value: white });

  return (
    <Link href={`/type/${type.type_id}`} className="block h-full">
      <Card className="h-full hover:border-foreground/30 transition-colors cursor-pointer">
        <CardContent className="p-4">
          {/* Header row: name + political lean */}
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="min-w-0">
              <span
                className="block text-[11px] uppercase tracking-wide mb-0.5"
                style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-sans)" }}
              >
                Type {type.type_id}
              </span>
              <span
                className="block text-[15px] font-bold leading-tight"
                style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
              >
                {displayName}
              </span>
            </div>
            <div className="flex-shrink-0 pt-0.5">
              <MarginDisplay demShare={type.mean_pred_dem_share} size="sm" />
            </div>
          </div>

          {/* Key demographics row */}
          {stats.length > 0 && (
            <div
              className="flex flex-wrap gap-x-3 gap-y-1 text-[12px]"
              style={{ color: "var(--color-text-muted)" }}
            >
              {stats.map((s) => (
                <span key={s.label}>
                  <span className="font-medium" style={{ color: "var(--color-text)" }}>
                    {s.value}
                  </span>{" "}
                  {s.label}
                </span>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
