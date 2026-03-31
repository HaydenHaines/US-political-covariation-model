import { formatMargin } from "@/lib/format";

// Grade colors follow 538/Silver Bulletin convention:
// A-tier = green, B-tier = blue, C-tier = amber, D/F/Banned = red
function gradeColor(grade: string): { bg: string; text: string; border: string } {
  const g = grade.replace(/[+-/]/g, "").charAt(0);
  switch (g) {
    case "A": return { bg: "rgba(34, 197, 94, 0.12)", text: "#15803d", border: "rgba(34, 197, 94, 0.35)" };
    case "B": return { bg: "rgba(59, 130, 246, 0.12)", text: "#1d4ed8", border: "rgba(59, 130, 246, 0.35)" };
    case "C": return { bg: "rgba(245, 158, 11, 0.12)", text: "#b45309", border: "rgba(245, 158, 11, 0.35)" };
    default:  return { bg: "rgba(239, 68, 68, 0.12)", text: "#b91c1c", border: "rgba(239, 68, 68, 0.35)" };
  }
}

function GradeBadge({ grade }: { grade: string }) {
  const { bg, text, border } = gradeColor(grade);
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 6px",
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 700,
        background: bg,
        color: text,
        border: `1px solid ${border}`,
        lineHeight: "18px",
      }}
      title={`Silver Bulletin grade: ${grade}`}
    >
      {grade}
    </span>
  );
}

export interface PollTableRow {
  date: string | null;
  pollster: string | null;
  dem_share: number;
  n_sample: number | null;
  grade: string | null;
}

interface PollTableProps {
  polls: PollTableRow[];
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
 * Static poll table — renders a list of polls server-side.
 *
 * Used in the race detail page for the SSR-fetched poll list.
 * For a live-updating version backed by SWR, use PollTracker.
 */
export function PollTable({ polls }: PollTableProps) {
  if (polls.length === 0) {
    return (
      <p className="text-sm italic" style={{ color: "var(--color-text-muted)" }}>
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
    <div className="overflow-x-auto">
      <table
        className="w-full text-sm border-collapse"
        aria-label="Race polls"
      >
        <thead>
          <tr
            className="border-b text-xs font-semibold uppercase tracking-wide"
            style={{
              borderColor: "var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            <th className="text-left py-2 pr-3">Date</th>
            <th className="text-left py-2 px-3">Pollster</th>
            <th className="text-center py-2 px-3">Grade</th>
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
                style={{ borderBottom: "1px solid var(--color-bg)" }}
              >
                <td
                  className="py-2 pr-3"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  {formatDate(poll.date)}
                </td>
                <td className="py-2 px-3">{poll.pollster ?? "Unknown"}</td>
                <td className="text-center py-2 px-3">
                  {poll.grade ? (
                    <GradeBadge grade={poll.grade} />
                  ) : (
                    <span style={{ color: "var(--color-text-muted)", fontSize: 12 }}>—</span>
                  )}
                </td>
                <td
                  className="text-right py-2 px-3 font-mono font-bold"
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
  );
}
