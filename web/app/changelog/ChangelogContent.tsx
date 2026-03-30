"use client";

import Link from "next/link";
import type { ChangelogResponse, ChangelogRaceDiff } from "@/lib/api";

function formatPct(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}

function formatDelta(delta: number): string {
  const pp = delta * 100;
  const sign = pp > 0 ? "+" : "";
  return `${sign}${pp.toFixed(1)}pp`;
}

function leanLabel(demShare: number): { text: string; color: string } {
  const margin = (demShare - 0.5) * 100;
  if (Math.abs(margin) < 0.5) return { text: "Toss-up", color: "var(--color-text-muted)" };
  const party = margin > 0 ? "D" : "R";
  const color = margin > 0 ? "var(--color-dem)" : "var(--color-gop)";
  return { text: `${party}+${Math.abs(margin).toFixed(1)}`, color };
}

function raceToSlug(race: string): string {
  return race.toLowerCase().replace(/ /g, "-");
}

function DiffRow({ diff }: { diff: ChangelogRaceDiff }) {
  const isNew = diff.before === null;
  const slug = raceToSlug(diff.race);
  const raceParts = diff.race.split(" ");
  const displayRace = raceParts.slice(1).join(" "); // drop year

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "8px 12px",
        borderBottom: "1px solid var(--color-border)",
        gap: 12,
      }}
    >
      <Link
        href={`/forecast/${slug}`}
        style={{
          textDecoration: "none",
          color: "var(--color-text)",
          fontWeight: 500,
          fontSize: 14,
          minWidth: 0,
        }}
      >
        {displayRace}
      </Link>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexShrink: 0,
          fontSize: 13,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {isNew && diff.after !== null ? (
          <>
            <span style={{ color: "var(--color-text-muted)" }}>new</span>
            <span style={{ fontWeight: 600, color: leanLabel(diff.after).color }}>
              {leanLabel(diff.after).text}
            </span>
          </>
        ) : diff.before !== null && diff.after !== null && diff.delta !== null ? (
          <>
            <span style={{ color: "var(--color-text-muted)" }}>
              {formatPct(diff.before)}
            </span>
            <span style={{ color: "var(--color-text-muted)" }}>&rarr;</span>
            <span style={{ fontWeight: 600, color: leanLabel(diff.after).color }}>
              {formatPct(diff.after)}
            </span>
            <span
              style={{
                fontSize: 12,
                fontWeight: 600,
                padding: "2px 6px",
                borderRadius: 4,
                background: diff.delta > 0 ? "rgba(33, 102, 172, 0.1)" : "rgba(215, 48, 39, 0.1)",
                color: diff.delta > 0 ? "var(--color-dem)" : "var(--color-gop)",
              }}
            >
              {formatDelta(diff.delta)}
            </span>
          </>
        ) : null}
      </div>
    </div>
  );
}

function ChangelogEntryCard({
  date,
  note,
  diffs,
}: {
  date: string;
  note: string | null;
  diffs: ChangelogRaceDiff[];
}) {
  const formattedDate = new Date(date + "T12:00:00").toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        overflow: "hidden",
        marginBottom: 24,
        background: "var(--color-surface, var(--color-bg))",
      }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid var(--color-border)",
          background: "var(--color-bg)",
        }}
      >
        <div
          style={{
            fontSize: 16,
            fontWeight: 700,
            fontFamily: "var(--font-serif)",
            color: "var(--color-text)",
          }}
        >
          {formattedDate}
        </div>
        {note && (
          <div
            style={{
              fontSize: 13,
              color: "var(--color-text-muted)",
              marginTop: 4,
            }}
          >
            {note}
          </div>
        )}
        <div
          style={{
            fontSize: 12,
            color: "var(--color-text-muted)",
            marginTop: 4,
          }}
        >
          {diffs.length} race{diffs.length !== 1 ? "s" : ""} updated
        </div>
      </div>
      <div>
        {diffs.map((d) => (
          <DiffRow key={d.race} diff={d} />
        ))}
      </div>
    </div>
  );
}

export function ChangelogContent({
  data,
}: {
  data: ChangelogResponse | null;
}) {
  return (
    <div
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: "48px 20px 80px",
      }}
    >
      {/* Header */}
      <nav
        aria-label="Breadcrumb"
        style={{ marginBottom: 24, fontSize: 13 }}
      >
        <ol
          style={{
            listStyle: "none",
            padding: 0,
            margin: 0,
            display: "flex",
            gap: 6,
          }}
        >
          <li>
            <Link
              href="/"
              style={{
                textDecoration: "none",
                color: "var(--color-text-muted)",
              }}
            >
              Home
            </Link>
          </li>
          <li style={{ color: "var(--color-text-muted)" }}>/</li>
          <li style={{ color: "var(--color-text)" }}>Forecast Changelog</li>
        </ol>
      </nav>

      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: 32,
          fontWeight: 700,
          margin: "0 0 8px",
          color: "var(--color-text)",
        }}
      >
        Forecast Changelog
      </h1>
      <p
        style={{
          fontSize: 16,
          lineHeight: 1.6,
          color: "var(--color-text-muted)",
          margin: "0 0 32px",
          maxWidth: 560,
        }}
      >
        How our forecasts have changed over time. Each entry shows races where
        the predicted Democratic two-party vote share shifted meaningfully after
        new polls or model updates.
      </p>

      {/* Entries */}
      {!data || data.entries.length === 0 ? (
        <div
          style={{
            padding: 32,
            textAlign: "center",
            color: "var(--color-text-muted)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
          }}
        >
          <p style={{ fontSize: 16, margin: "0 0 8px" }}>
            No forecast changes recorded yet.
          </p>
          <p style={{ fontSize: 14 }}>
            Changes will appear here after the next weekly poll scrape updates
            predictions.
          </p>
        </div>
      ) : (
        data.entries.map((entry) => (
          <ChangelogEntryCard
            key={entry.date}
            date={entry.date}
            note={entry.note}
            diffs={entry.diffs}
          />
        ))
      )}

      {/* Footer note */}
      <p
        style={{
          fontSize: 13,
          color: "var(--color-text-muted)",
          marginTop: 32,
          lineHeight: 1.6,
        }}
      >
        Snapshots are recorded after each weekly poll scrape (Sundays). Only
        races with poll-adjusted predictions are tracked. Changes under 0.2
        percentage points are filtered out. View{" "}
        <Link
          href="/methodology"
          style={{ color: "var(--color-dem)", textDecoration: "none" }}
        >
          our methodology
        </Link>{" "}
        for details on how predictions are computed.
      </p>
    </div>
  );
}
