import { cn } from "@/lib/utils";

interface FreshnessStampProps {
  updatedAt?: string | Date;
  pollCount?: number;
  className?: string;
}

/** Format a date as "Mar 23, 2026" — absolute dates age better than relative ones. */
function formatAbsoluteDate(date: string | Date): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function FreshnessStamp({ updatedAt, pollCount, className }: FreshnessStampProps) {
  const parts: string[] = [];

  if (updatedAt) {
    parts.push(`Updated ${formatAbsoluteDate(updatedAt)}`);
  }

  if (pollCount !== undefined) {
    parts.push(`${pollCount} poll${pollCount !== 1 ? "s" : ""}`);
  }

  if (parts.length === 0) return null;

  return (
    <span className={cn("text-sm text-muted-foreground", className)}>
      {parts.join(" · ")}
    </span>
  );
}
