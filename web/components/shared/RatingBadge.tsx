"use client";

import { Badge } from "@/components/ui/badge";
import { RATING_COLORS, RATING_LABELS } from "@/lib/config/palette";

interface RatingBadgeProps {
  rating: string;
  className?: string;
}

export function RatingBadge({ rating, className }: RatingBadgeProps) {
  const color = RATING_COLORS[rating as keyof typeof RATING_COLORS] ?? RATING_COLORS.tossup;
  const label = RATING_LABELS[rating as keyof typeof RATING_LABELS] ?? rating;

  return (
    <Badge
      className={className}
      style={{ backgroundColor: color, color: "#fff", border: "none" }}
    >
      {label}
    </Badge>
  );
}
