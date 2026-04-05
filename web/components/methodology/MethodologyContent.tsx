"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";

// ── Static data ────────────────────────────────────────────────────────────

const TOC_SECTIONS = [
  { id: "key-insight", label: "The Key Insight" },
  { id: "how-it-works", label: "How It Works" },
  { id: "performance", label: "Model Performance" },
  { id: "historical-accuracy", label: "Historical Accuracy" },
  { id: "backtest-validation", label: "2022 Backtest" },
  { id: "differentiation", label: "What Makes This Different" },
  { id: "data-sources", label: "Data Sources" },
  { id: "status", label: "Current Status" },
  { id: "credits", label: "Credits" },
] as const;

type SectionId = (typeof TOC_SECTIONS)[number]["id"];

const MODEL_METRICS = [
  { label: "LOO r (Ensemble)", value: "0.711", note: "Ridge + HGB, 160 features" },
  { label: "LOO r (Ridge)", value: "0.533", note: "Type scores + county mean" },
  { label: "Holdout r", value: "0.698", note: "Standard hold-out validation" },
  { label: "Coherence", value: "0.783", note: "Within-type political agreement" },
  { label: "RMSE", value: "0.073", note: "Root mean squared error" },
  { label: "Covariance Val r", value: "0.915", note: "Ledoit-Wolf regularized" },
  { label: "Counties", value: "3,154", note: "All 50 states + DC" },
  { label: "Types", value: "100", note: "KMeans discovered" },
];

const CROSS_ELECTION = [
  { cycle: "2012 → 2016", r: 0.52 },
  { cycle: "2016 → 2020", r: 0.55 },
  { cycle: "2020 → 2024", r: 0.38 },
  { cycle: "2008 → 2012", r: 0.43 },
];

// Backtest results: model retrained without 2022 data, predicting 2022 outcomes
// Source: data/experiments/backtest_2022_results.json
const BACKTEST_METRICS = [
  {
    race: "Senate",
    countyR: 0.9705,
    countyRmse: 0.0401,
    nCounties: 1880,
    typeMeanR: 0.6789,
  },
  {
    race: "Governor",
    countyR: 0.8856,
    countyRmse: 0.0825,
    nCounties: 1900,
    typeMeanR: 0.6359,
  },
] as const;

// Key spotlight races for the Senate backtest — best illustration of r=0.97 precision
const SENATE_SPOTLIGHT = [
  { state: "AZ", label: "Kelly vs Masters", pred: 0.5005, actual: 0.525, error: -2.45 },
  { state: "GA", label: "Warnock vs Walker", pred: 0.4988, actual: 0.5049, error: -0.61 },
  { state: "NC", label: "Beasley vs Budd", pred: 0.4928, actual: 0.4835, error: 0.93 },
  { state: "NV", label: "Cortez Masto vs Laxalt", pred: 0.5092, actual: 0.504, error: 0.52 },
  { state: "OH", label: "Ryan vs Vance", pred: 0.4564, actual: 0.4694, error: -1.3 },
  { state: "PA", label: "Fetterman vs Oz", pred: 0.501, actual: 0.5252, error: -2.41 },
  { state: "WI", label: "Barnes vs Johnson", pred: 0.5034, actual: 0.495, error: 0.84 },
] as const;

const DATA_SOURCES = [
  { name: "Election returns", source: "MIT Election Data & Science Lab (MEDSL)" },
  { name: "Demographics", source: "US Census Bureau — Decennial 2000/2010/2020 + ACS 5-year" },
  { name: "Religious congregations", source: "ARDA — Religious Congregations & Membership Study (RCMS 2020)" },
  { name: "Industry composition", source: "BLS Quarterly Census of Employment and Wages (QCEW)" },
  { name: "Health behaviors", source: "County Health Rankings (Robert Wood Johnson Foundation)" },
  { name: "Migration flows", source: "IRS Statistics of Income (SOI) — county-to-county migration" },
  { name: "Social connectivity", source: "Facebook Social Connectedness Index (county-pair network)" },
  { name: "Broadband access", source: "FCC / ACS — internet subscription at county level" },
  { name: "Polling data", source: "FiveThirtyEight archives + Silver Bulletin pollster ratings" },
  { name: "Governor returns", source: "Algara & Amlani (Harvard Dataverse) — 2002-2022 governor" },
];

// ── Sticky TOC sidebar ─────────────────────────────────────────────────────

function TableOfContents({ activeId }: { activeId: SectionId }) {
  return (
    <nav
      aria-label="Table of contents"
      className="hidden lg:block sticky top-20 self-start w-52 shrink-0"
    >
      <p
        className="text-xs font-bold uppercase tracking-widest px-3 pb-2"
        style={{ color: "var(--color-text-muted)" }}
      >
        On this page
      </p>
      <ul className="space-y-0.5">
        {TOC_SECTIONS.map((section) => (
          <li key={section.id}>
            <a
              href={`#${section.id}`}
              className="block px-3 py-1.5 text-sm rounded transition-colors"
              style={{
                color: activeId === section.id ? "var(--color-dem)" : "var(--color-text-muted)",
                background: activeId === section.id ? "var(--color-surface)" : "transparent",
                fontWeight: activeId === section.id ? "600" : "400",
                textDecoration: "none",
                borderLeft: activeId === section.id
                  ? "2px solid var(--color-dem)"
                  : "2px solid transparent",
              }}
            >
              {section.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

// ── ScrollSection ──────────────────────────────────────────────────────────

/**
 * A full-height section. Always expanded — no collapse toggle — so TOC links
 * always land on visible content, matching the scrollytelling design intent.
 */
function ScrollSection({
  id,
  title,
  sectionNumber,
  totalSections,
  children,
}: {
  id: string;
  title: string;
  sectionNumber: number;
  totalSections: number;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="scroll-mt-16"
      style={{
        minHeight: "80vh",
        paddingTop: "3rem",
        paddingBottom: "3rem",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        borderBottom: sectionNumber < totalSections ? "1px solid var(--color-border)" : "none",
      }}
    >
      {/* Section header */}
      <div className="flex items-center gap-3 min-w-0 mb-5 group">
        <span
          aria-hidden="true"
          className="shrink-0 text-xs font-mono tabular-nums select-none"
          style={{
            color: "var(--color-text-subtle)",
            fontFamily: "var(--font-sans)",
            letterSpacing: "0.05em",
          }}
        >
          {String(sectionNumber).padStart(2, "0")} / {String(totalSections).padStart(2, "0")}
        </span>

        <h2
          className="font-serif text-2xl font-bold leading-snug flex items-center gap-2"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          {title}
          <a
            href={`#${id}`}
            className="text-sm font-normal opacity-0 group-hover:opacity-60 transition-opacity"
            style={{ color: "var(--color-text-muted)", textDecoration: "none" }}
            aria-label={`Link to ${title} section`}
          >
            #
          </a>
        </h2>
      </div>

      <div
        id={`${id}-body`}
        role="region"
        aria-labelledby={id}
        className="text-base leading-relaxed"
        style={{ color: "var(--color-text)" }}
      >
        {children}
      </div>

      <div
        className="mt-auto pt-8 flex justify-center"
        aria-hidden="true"
        style={{ opacity: 0.25 }}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-5 h-5" style={{ color: "var(--color-text-muted)" }}>
          <path d="M12 5v14M5 12l7 7 7-7" />
        </svg>
      </div>
    </section>
  );
}

// ── Step (numbered, scroll-prominent) ─────────────────────────────────────

/**
 * Individual step inside "How It Works". The number circle is prominent and
 * sits in the left margin so it reads as a visual anchor while scrolling.
 */
function Step({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="flex gap-5 mb-10"
      style={{ scrollMarginTop: "5rem" }}
    >
      {/* Large step number — Dusty Ink palette */}
      <div className="shrink-0 flex flex-col items-center" style={{ paddingTop: "2px" }}>
        <span
          aria-hidden="true"
          className="inline-flex items-center justify-center w-10 h-10 rounded-full text-base font-bold"
          style={{
            background: "var(--forecast-safe-d)",
            color: "#ffffff",
            fontFamily: "var(--font-sans)",
            boxShadow: "0 2px 6px rgba(45,74,111,0.25)",
          }}
        >
          {number}
        </span>
        {/* Vertical connector line (except last step) */}
        <div
          className="flex-1 mt-2"
          style={{
            width: "1px",
            minHeight: "24px",
            background: "var(--color-border)",
          }}
        />
      </div>

      {/* Step content */}
      <div className="min-w-0 flex-1 pb-2">
        <h3
          className="font-serif text-lg font-semibold mb-2"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Step {number}: {title}
        </h3>
        <div className="space-y-3 text-base leading-relaxed" style={{ color: "var(--color-text)" }}>
          {children}
        </div>
      </div>
    </div>
  );
}

// ── MetricGrid ─────────────────────────────────────────────────────────────

function MetricGrid({
  metrics,
}: {
  metrics: { label: string; value: string; note?: string }[];
}) {
  return (
    <div
      role="list"
      aria-label="Model performance metrics"
      className="grid gap-3 mt-4 mb-2"
      style={{ gridTemplateColumns: "repeat(auto-fit, minmax(175px, 1fr))" }}
    >
      {metrics.map((m) => (
        <div
          key={m.label}
          role="listitem"
          className="rounded-md p-4"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
          }}
        >
          <div
            className="text-xs font-semibold uppercase tracking-wider mb-1"
            style={{ color: "var(--color-text-muted)" }}
          >
            {m.label}
          </div>
          <div
            className="font-serif text-2xl font-bold leading-none mb-1"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            {m.value}
          </div>
          {m.note && (
            <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              {m.note}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── CrossElectionTable ─────────────────────────────────────────────────────

function CrossElectionTable({
  rows,
}: {
  rows: { cycle: string; r: number }[];
}) {
  const maxR = Math.max(...rows.map((r) => r.r));

  return (
    <div
      className="rounded-md overflow-hidden mt-4"
      style={{ border: "1px solid var(--color-border)" }}
      role="table"
      aria-label="Cross-election LOO r results"
    >
      <div
        className="grid grid-cols-3 px-4 py-2 text-xs font-semibold uppercase tracking-wider"
        role="row"
        style={{
          background: "var(--color-surface)",
          color: "var(--color-text-muted)",
          borderBottom: "1px solid var(--color-border)",
        }}
      >
        <span role="columnheader">Election Cycle</span>
        <span role="columnheader">LOO r</span>
        <span role="columnheader" aria-hidden="true" />
      </div>
      {rows.map((row) => (
        <div
          key={row.cycle}
          role="row"
          className="grid grid-cols-3 items-center px-4 py-3"
          style={{ borderBottom: "1px solid var(--color-border)" }}
        >
          <span role="cell" className="text-sm font-medium">{row.cycle}</span>
          <span role="cell" className="text-sm font-mono font-semibold">{row.r.toFixed(2)}</span>
          <div role="cell" className="h-2 rounded-full overflow-hidden" style={{ background: "var(--color-border)" }}>
            <div
              className="h-full rounded-full"
              style={{
                width: `${(row.r / maxR) * 100}%`,
                background: "var(--color-dem)",
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── BacktestSummaryCards ───────────────────────────────────────────────────

/**
 * Side-by-side cards showing the two headline backtest metrics:
 * Senate county r and Governor county r from the 2022 holdout.
 * The type-mean baseline is shown for contrast — it illustrates how much
 * the presidential-prior enrichment adds beyond just using type averages.
 */
function BacktestSummaryCards() {
  return (
    <div
      role="list"
      aria-label="2022 backtest summary metrics"
      className="grid gap-4 mt-4 mb-2"
      style={{ gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))" }}
    >
      {BACKTEST_METRICS.map((m) => (
        <div
          key={m.race}
          role="listitem"
          className="rounded-md p-5"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
          }}
        >
          <div
            className="text-xs font-bold uppercase tracking-wider mb-3"
            style={{ color: "var(--color-text-muted)" }}
          >
            {m.race} races — {m.nCounties.toLocaleString()} counties
          </div>

          {/* Primary metric: presidential-prior county r */}
          <div className="flex items-end gap-3 mb-3">
            <div>
              <div
                className="font-serif text-4xl font-bold leading-none"
                style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
              >
                {m.countyR.toFixed(4)}
              </div>
              <div className="text-xs mt-1" style={{ color: "var(--color-text-muted)" }}>
                county r — presidential prior
              </div>
            </div>
          </div>

          {/* Progress bar — county r */}
          <div
            className="h-2 rounded-full overflow-hidden mb-4"
            style={{ background: "var(--color-border)" }}
            aria-hidden="true"
          >
            <div
              className="h-full rounded-full"
              style={{
                width: `${m.countyR * 100}%`,
                background: "var(--color-dem)",
              }}
            />
          </div>

          {/* Comparison: type-mean baseline */}
          <div
            className="pt-3 flex justify-between items-center text-sm"
            style={{ borderTop: "1px solid var(--color-border)" }}
          >
            <span style={{ color: "var(--color-text-muted)" }}>Type-mean baseline</span>
            <span
              className="font-mono font-semibold"
              style={{ color: "var(--color-text-muted)" }}
            >
              r = {m.typeMeanR.toFixed(4)}
            </span>
          </div>

          {/* RMSE */}
          <div
            className="pt-2 flex justify-between items-center text-sm"
          >
            <span style={{ color: "var(--color-text-muted)" }}>RMSE</span>
            <span
              className="font-mono font-semibold"
              style={{ color: "var(--color-text-muted)" }}
            >
              {m.countyRmse.toFixed(4)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── BacktestSpotlightTable ─────────────────────────────────────────────────

/**
 * Shows predicted vs actual Dem share for seven competitive 2022 Senate races.
 * These are the races the model had never seen when it made its predictions —
 * they were held out of training. The small errors here (±3pp for most)
 * are what the r=0.97 translates to in practice.
 */
function BacktestSpotlightTable() {
  return (
    <div
      className="rounded-md overflow-hidden mt-4"
      style={{ border: "1px solid var(--color-border)" }}
      role="table"
      aria-label="2022 Senate backtest spotlight races"
    >
      {/* Header */}
      <div
        className="grid px-4 py-2 text-xs font-semibold uppercase tracking-wider"
        role="row"
        style={{
          gridTemplateColumns: "5rem 1fr 5rem 5rem 5rem",
          background: "var(--color-surface)",
          color: "var(--color-text-muted)",
          borderBottom: "1px solid var(--color-border)",
        }}
      >
        <span role="columnheader">State</span>
        <span role="columnheader">Race</span>
        <span role="columnheader" className="text-right">Predicted</span>
        <span role="columnheader" className="text-right">Actual</span>
        <span role="columnheader" className="text-right">Error</span>
      </div>

      {/* Rows */}
      {SENATE_SPOTLIGHT.map((row, i) => {
        const absErr = Math.abs(row.error);
        // Color-code errors: ≤2pp green-ish, ≤4pp neutral, >4pp orange-ish
        const errColor =
          absErr <= 2
            ? "var(--color-dem)"
            : absErr <= 4
            ? "var(--color-text)"
            : "var(--color-rep)";

        return (
          <div
            key={row.state}
            role="row"
            className="grid items-center px-4 py-3 text-sm"
            style={{
              gridTemplateColumns: "5rem 1fr 5rem 5rem 5rem",
              borderBottom:
                i < SENATE_SPOTLIGHT.length - 1
                  ? "1px solid var(--color-border)"
                  : "none",
              background: i % 2 === 0 ? "transparent" : "var(--color-surface)",
            }}
          >
            <span role="cell" className="font-bold font-mono">{row.state}</span>
            <span role="cell" style={{ color: "var(--color-text-muted)" }}>{row.label}</span>
            <span role="cell" className="text-right font-mono">{(row.pred * 100).toFixed(1)}%</span>
            <span role="cell" className="text-right font-mono">{(row.actual * 100).toFixed(1)}%</span>
            <span role="cell" className="text-right font-mono font-semibold" style={{ color: errColor }}>
              {row.error > 0 ? "+" : ""}{row.error.toFixed(1)}pp
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export function MethodologyContent() {
  const [activeSection, setActiveSection] = useState<SectionId>("key-insight");
  const observerRef = useRef<IntersectionObserver | null>(null);

  const setupObserver = useCallback(() => {
    observerRef.current?.disconnect();

    observerRef.current = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length > 0) {
          setActiveSection(visible[0].target.id as SectionId);
        }
      },
      // Large top margin so the section activates as it enters the upper third
      { rootMargin: "-10% 0px -55% 0px", threshold: 0 },
    );

    for (const section of TOC_SECTIONS) {
      const el = document.getElementById(section.id);
      if (el) observerRef.current.observe(el);
    }

    return () => observerRef.current?.disconnect();
  }, []);

  useEffect(() => {
    const cleanup = setupObserver();
    return cleanup;
  }, [setupObserver]);

  const total = TOC_SECTIONS.length;

  return (
    <div className="max-w-5xl mx-auto px-4 pb-20 flex gap-12">
        {/* Sticky TOC sidebar */}
        <TableOfContents activeId={activeSection} />

        {/* Main content */}
        <div className="min-w-0 flex-1">
          {/* Breadcrumb */}
          <nav aria-label="breadcrumb" className="text-xs mb-6" style={{ color: "var(--color-text-muted)" }}>
            <ol className="flex flex-wrap items-center gap-x-1 list-none p-0 m-0">
              <li>
                <Link
                  href="/"
                  style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
                >
                  Home
                </Link>
              </li>
              <li aria-hidden="true">/</li>
              <li aria-current="page">Methodology</li>
            </ol>
          </nav>

          {/* Page hero — not a scroll section, just an intro header */}
          <header className="mb-8 pt-2">
            <h1
              className="font-serif text-4xl font-bold leading-tight mb-4"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              How WetherVane Works
            </h1>
            <p
              className="text-lg leading-relaxed pl-5"
              style={{
                borderLeft: "3px solid var(--color-border)",
                color: "var(--color-text)",
              }}
            >
              WetherVane discovers communities of voters who move together
              politically — then uses that structure to propagate new information
              across geography. A poll in one state updates predictions in every
              state that shares those communities.
            </p>
          </header>

          {/* ── 01 — The Key Insight ─────────────────────────── */}
          <ScrollSection
            id="key-insight"
            title="The Key Insight"
            sectionNumber={1}
            totalSections={total}
          >
            <p>
              Most forecasting models treat counties as independent: average the
              polls, adjust for house effects, output a number. WetherVane starts
              from a different question:{" "}
              <em>what is the underlying structure that makes places move together?</em>
            </p>
            <p className="mt-3">
              Thousands of places share hidden behavioral patterns. A rural
              evangelical county in Georgia moves with rural evangelical counties
              in Iowa — not because anyone coordinates, but because the same
              forces act on similar communities. Discovering that structure —
              not just reading the surface results — is what makes prediction
              defensible.
            </p>
            <p className="mt-3">
              The model is built for readers who want to understand electoral
              dynamics, not just consume a top-line forecast. If you read
              FiveThirtyEight for the methodology write-ups, this is for you.
            </p>
          </ScrollSection>

          {/* ── 02 — How It Works ───────────────────────────── */}
          <ScrollSection
            id="how-it-works"
            title="How It Works"
            sectionNumber={2}
            totalSections={total}
          >
            <Step number={1} title="Measure Shifts">
              <p>
                The model begins by computing how every county shifted politically
                across each pair of elections from 2008 to 2024. These shift
                vectors capture direction and magnitude — did this county swing
                toward Democrats or Republicans, and by how much?
              </p>
            </Step>

            <Step number={2} title="Discover Types (KMeans J=100)">
              <p>
                KMeans clustering (<strong>J=100</strong>) groups counties with
                similar shift patterns into <strong>electoral types</strong>.
                Presidential shifts are weighted <strong>8×</strong> because they
                carry cross-state signal. Governor and Senate shifts are{" "}
                <em>state-centered</em> first — subtracting the statewide swing —
                so clustering captures within-state variation, not just
                red-state/blue-state geography.
              </p>
              <p>
                The result is 100 fine-grained electoral types and 5{" "}
                <strong>super-types</strong> (broad behavioral families), which
                form the colors of the stained glass map.
              </p>
            </Step>

            <Step number={3} title="Map Soft Membership">
              <p>
                No county belongs to just one type. Each county has partial
                membership in multiple types, computed via temperature-scaled
                inverse distance in shift space (temperature <strong>T=10</strong>).
                A suburban Atlanta county might be 40% &quot;College-Educated
                Suburban&quot; and 30% &quot;Black Belt &amp; Diverse.&quot;
              </p>
              <p>
                Soft membership reduces calibration error by ~37% compared to
                hard assignment. The map color reflects dominant type; predictions
                use the full membership vector.
              </p>
            </Step>

            <Step number={4} title="Estimate Covariance (Ledoit-Wolf)">
              <p>
                Types that share electoral behavior tend to co-move. The model
                estimates a 100×100 covariance matrix capturing how much each
                pair of types correlates, using observed electoral correlation
                with Ledoit-Wolf regularization (validation r = <strong>0.915</strong>).
              </p>
              <p>
                This covariance structure encodes which types are behaviorally
                coupled — and therefore how information should flow between them
                when a new poll arrives.
              </p>
            </Step>

            <Step number={5} title="Propagate Polls">
              <p>
                When a new poll arrives, the model uses a{" "}
                <strong>Bayesian Gaussian (Kalman filter) update</strong> — exact
                and closed-form, no simulation needed. Multiple polls stack as
                independent observations. Because types cross state lines, a
                Florida Senate poll shifts Georgia predictions too.
              </p>
              <p>
                The final ensemble uses{" "}
                <strong>Ridge + Histogram Gradient Boosting</strong> with 160
                features from 8 independent data sources, achieving a
                leave-one-out r of <strong>0.711</strong>.
              </p>
            </Step>
          </ScrollSection>

          {/* ── 03 — Model Performance ──────────────────────── */}
          <ScrollSection
            id="performance"
            title="Model Performance"
            sectionNumber={3}
            totalSections={total}
          >
            <p>
              All metrics are on the 2024 presidential election. LOO
              (leave-one-out) cross-validation excludes each county from its own
              type mean before predicting it — this is the honest generalization
              metric that cannot be inflated by self-prediction.
            </p>
            <MetricGrid metrics={MODEL_METRICS} />
            <p className="text-sm mt-3" style={{ color: "var(--color-text-muted)" }}>
              The standard holdout r (0.698) is inflated by ~0.22 because
              counties help predict their own type means. LOO r (0.711) is the
              correct metric for evaluating generalization. Both are reported
              for transparency.
            </p>
          </ScrollSection>

          {/* ── 04 — Historical Accuracy ────────────────────── */}
          <ScrollSection
            id="historical-accuracy"
            title="Historical Accuracy"
            sectionNumber={4}
            totalSections={total}
          >
            <p>
              Cross-election validation tests whether the type structure
              discovered from historical shifts generalizes to new cycles.
              Across four presidential election pairs, mean LOO r ={" "}
              <strong>0.476 ± 0.10</strong>.
            </p>
            <CrossElectionTable rows={CROSS_ELECTION} />
            <p className="mt-4">
              Not all cycles are equally predictable.{" "}
              <strong>2020→2024 (r=0.38)</strong> was the hardest — the
              Harris-Trump dynamic produced unusual cross-type movement,
              particularly among Hispanic communities. The{" "}
              <strong>2012→2016 transition (r=0.52)</strong> was most
              predictable: Trump&apos;s initial surge followed existing type
              fault lines closely.
            </p>
            <div className="mt-4">
              <Link
                href="/methodology/accuracy"
                className="text-sm font-semibold"
                style={{ color: "var(--color-dem)", textDecoration: "none" }}
              >
                View full backtesting results →
              </Link>
            </div>
          </ScrollSection>

          {/* ── 05 — 2022 Backtest Validation ──────────────── */}
          <ScrollSection
            id="backtest-validation"
            title="2022 Backtest Validation"
            sectionNumber={5}
            totalSections={total}
          >
            <p>
              The cross-election LOO metrics above measure whether the{" "}
              <em>type structure</em> generalizes to new presidential cycles. But
              the harder test is this: does the model generalize to{" "}
              <strong>a completely unseen election type</strong> — a midterm it
              has never trained on?
            </p>
            <p className="mt-3">
              To test this, we retrained the model from scratch after{" "}
              <strong>removing all 2022 data</strong> — no 2022 governor shifts,
              no 2022 Senate shifts, none of it. We then used the 2020
              presidential Dem share as a county-level prior, and predicted 2022
              outcomes purely from the structure the model had learned through 2020.
            </p>
            <BacktestSummaryCards />
            <p className="mt-4">
              The Senate result — <strong>county r = 0.97</strong> across 1,880
              counties — is the strongest evidence that the partisan geography
              structure the model learns is real and durable. Correlations this
              high mean the model gets the <em>relative</em> ordering of counties
              nearly perfect: which ones lean Democratic, which lean Republican,
              and by roughly how much. It does not mean point estimates are
              perfect — the RMSE of ±4pp on Senate races reflects real
              uncertainty about national environment and candidate effects.
            </p>
            <p className="mt-3">
              Governor races are harder (r = 0.89) because governor outcomes
              depend more heavily on candidate-specific factors and state-level
              dynamics that a national type model cannot fully capture. The
              type structure still explains 79% of the variance in county-level
              governor outcomes — but the remaining 21% is genuine candidate
              effect.
            </p>

            <h3
              className="font-serif text-lg font-semibold mt-6 mb-2"
              style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
            >
              Seven competitive Senate races, never seen during training
            </h3>
            <p className="mb-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
              The model predicted these outcomes using only the type structure
              learned from 2008–2020. Errors are predicted minus actual
              Democratic share (state-level, vote-weighted average across counties).
            </p>
            <BacktestSpotlightTable />
            <p className="mt-4 text-sm" style={{ color: "var(--color-text-muted)" }}>
              Five of seven races land within ±2.5pp. The type-mean baseline
              (r = 0.68) predicts these same races with errors up to ±17pp —
              showing that the presidential-prior enrichment is doing substantial
              work beyond just knowing which communities lean which way.
            </p>
          </ScrollSection>

          {/* ── 06 — What Makes This Different ─────────────── */}
          <ScrollSection
            id="differentiation"
            title="What Makes This Different"
            sectionNumber={6}
            totalSections={total}
          >
            <ul className="space-y-4 pl-5 list-disc">
              <li>
                <strong>Structure from behavior, not demographics.</strong> Types
                are discovered from how places shift electorally. Demographics
                describe the types after discovery — they do not define them. This
                avoids baking in assumptions about which demographic groups drive
                politics.
              </li>
              <li>
                <strong>Cross-state information sharing.</strong> Because types
                cross state lines, a poll in one state informs predictions in
                another. Most models treat states as independent. WetherVane
                treats the country as one connected landscape.
              </li>
              <li>
                <strong>Full uncertainty quantification.</strong> Every prediction
                comes with 90% credible intervals. Intervals widen where the model
                has less data and tighten where type signals are strong.
              </li>
              <li>
                <strong>Transparent and interpretable.</strong> Every prediction
                traces back to specific types, their shift patterns, and the polls
                that influenced them. Not a black box — inspect it on{" "}
                <Link
                  href="/forecast"
                  style={{ color: "var(--color-dem)", textDecoration: "none" }}
                >
                  the map
                </Link>
                .
              </li>
              <li>
                <strong>Free data only.</strong> No proprietary datasets, no paid
                subscriptions. Every source listed below is publicly available,
                making the model fully reproducible.
              </li>
            </ul>
          </ScrollSection>

          {/* ── 07 — Data Sources ───────────────────────────── */}
          <ScrollSection
            id="data-sources"
            title="Data Sources"
            sectionNumber={7}
            totalSections={total}
          >
            <p>
              WetherVane uses exclusively free, public data. No proprietary
              datasets or paid subscriptions.
            </p>
            <div
              className="mt-3 rounded-md overflow-hidden"
              style={{ border: "1px solid var(--color-border)" }}
              role="list"
              aria-label="Data sources"
            >
              {DATA_SOURCES.map((row, i) => (
                <div
                  key={row.name}
                  role="listitem"
                  className="flex justify-between items-baseline gap-4 px-4 py-3 flex-wrap"
                  style={{
                    borderBottom:
                      i < DATA_SOURCES.length - 1
                        ? "1px solid var(--color-border)"
                        : "none",
                    background: i % 2 === 0 ? "transparent" : "var(--color-surface)",
                  }}
                >
                  <span className="font-semibold text-sm shrink-0">{row.name}</span>
                  <span className="text-sm text-right" style={{ color: "var(--color-text-muted)" }}>
                    {row.source}
                  </span>
                </div>
              ))}
            </div>
          </ScrollSection>

          {/* ── 08 — Current Status ─────────────────────────── */}
          <ScrollSection
            id="status"
            title="Current Status"
            sectionNumber={8}
            totalSections={total}
          >
            <p>
              WetherVane is in active development, targeting the{" "}
              <strong>2026 midterm elections</strong>. The model currently covers
              all 50 states and DC, tracking 33 Senate races.
            </p>
            <p className="mt-3">
              The poll scraper runs weekly, ingesting new polls and updating race
              forecasts automatically. Individual race forecasts are available on
              the{" "}
              <Link
                href="/forecast"
                style={{ color: "var(--color-dem)", textDecoration: "none" }}
              >
                forecast page
              </Link>
              .
            </p>
            <div
              className="mt-4 grid gap-3 text-sm"
              style={{ gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))" }}
            >
              {[
                { label: "Counties", value: "3,154" },
                { label: "Electoral types", value: "100" },
                { label: "Super-types", value: "5" },
                { label: "Races tracked", value: "33" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-md px-4 py-3 text-center"
                  style={{
                    background: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                  }}
                >
                  <div
                    className="font-serif text-2xl font-bold mb-1"
                    style={{ fontFamily: "var(--font-serif)" }}
                  >
                    {stat.value}
                  </div>
                  <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-4">
              <strong>Planned improvements:</strong> BEA regional economic data,
              FEC donor density features, richer poll ingestion with crosstab
              disaggregation — crosstabs tell us which types were sampled, so a
              poll oversampling college-educated voters should pull harder on
              types with high college-educated membership.
            </p>
          </ScrollSection>

          {/* ── 09 — Credits ────────────────────────────────── */}
          <ScrollSection
            id="credits"
            title="Credits"
            sectionNumber={9}
            totalSections={total}
          >
            <p>
              Built by <strong>Hayden Haines</strong>.
            </p>
            <p className="text-sm mt-3" style={{ color: "var(--color-text-muted)" }}>
              Methodology inspired by The Economist&apos;s 2020 presidential model
              (Heidemanns, Gelman &amp; Morris). Type-covariance architecture
              adapted to shift-based community discovery.
            </p>
            <p className="text-sm mt-2" style={{ color: "var(--color-text-muted)" }}>
              Election return data from MIT MEDSL and Algara &amp; Amlani (Harvard
              Dataverse). All other data sources are listed above.
            </p>
          </ScrollSection>

          {/* Footer nav */}
          <div
            className="pt-6 flex justify-between items-center flex-wrap gap-3"
            style={{ borderTop: "1px solid var(--color-border)" }}
          >
            <Link
              href="/"
              className="text-sm font-semibold"
              style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
            >
              ← Home
            </Link>
            <Link
              href="/forecast"
              className="text-sm font-semibold"
              style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}
            >
              View 2026 race forecasts →
            </Link>
          </div>
        </div>
    </div>
  );
}
