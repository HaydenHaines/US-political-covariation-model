import type { CSSProperties } from "react";
import { parseMargin } from "@/lib/format";
import { cn } from "@/lib/utils";

type Size = "sm" | "md" | "lg" | "xl";

interface MarginDisplayProps {
  demShare: number | null;
  size?: Size;
  className?: string;
}

const SIZE_CLASSES: Record<Size, string> = {
  sm: "text-sm",
  md: "text-lg font-semibold",
  lg: "text-2xl font-bold",
  xl: "text-5xl font-bold tracking-tight",
};

const PARTY_STYLES: Record<"dem" | "gop" | "even", CSSProperties> = {
  dem:  { color: "var(--forecast-safe-d)" },
  gop:  { color: "var(--forecast-safe-r)" },
  even: { color: "var(--forecast-tossup)" },
};

export function MarginDisplay({ demShare, size = "md", className }: MarginDisplayProps) {
  const { text, party } = parseMargin(demShare);

  return (
    <span
      className={cn(SIZE_CLASSES[size], "font-mono", className)}
      style={PARTY_STYLES[party]}
    >
      {text}
    </span>
  );
}
