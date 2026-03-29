import { timeAgo } from "@/lib/format";
import { cn } from "@/lib/utils";

interface FreshnessStampProps {
  updatedAt?: string | Date;
  pollCount?: number;
  className?: string;
}

export function FreshnessStamp({ updatedAt, pollCount, className }: FreshnessStampProps) {
  const parts: string[] = [];

  if (updatedAt) {
    parts.push(`Updated ${timeAgo(updatedAt)}`);
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
