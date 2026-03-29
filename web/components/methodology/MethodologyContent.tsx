"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";

// ── Static data ────────────────────────────────────────────────────────────

const TOC_SECTIONS = [
  { id: "key-insight", label: "The Key Insight" },
  { id: "how-it-works", label: "How It Works" },
  { id: "performance", label: "Model Performance" },
  { id: "historical-accuracy", label: "Historical Accuracy" },
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

// ── Sub-components ─────────────────────────────────────────────────────────

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

function ExpandableSection({
  id,
  title,
  defaultExpanded = true,
  children,
}: {
  id: string;
  title: string;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <section id={id} className="mb-10 scroll-mt-20">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-controls={`${id}-body`}
        className="flex w-full items-center justify-between gap-3 text-left mb-4 group"
      >
        <h2
          className="font-serif text-2xl font-bold leading-snug flex items-center gap-2"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          {title}
          <a
            href={`#${id}`}
            className="text-sm font-normal opacity-0 group-hover:opacity-60 transition-opacity"
            style={{ color: "var(--color-text-muted)", textDecoration: "none" }}
            onClick={(e) => e.stopPropagation()}
            aria-label={`Link to ${title} section`}
          >
            #
          </a>
        </h2>
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          className="w-5 h-5 shrink-0 transition-transform"
          style={{
            transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
            color: "var(--color-text-muted)",
          }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      <div
        id={`${id}-body`}
        role="region"
        aria-labelledby={id}
        className="text-base leading-relaxed overflow-hidden transition-all"
        style={{
          display: expanded ? "block" : "none",
          color: "var(--color-text)",
        }}
      >
        {children}
      </div>
    </section>
  );
}

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
    <div className="mb-7">
      <h3
        className="font-serif text-lg font-semibold flex items-center gap-3 mb-2"
        style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
      >
        <span
          aria-hidden="true"
          className="inline-flex items-center justify-center shrink-0 w-7 h-7 rounded-full text-xs font-bold"
          style={{
            background: "var(--color-text)",
            color: "var(--color-surface)",
            fontFamily: "var(--font-sans)",
          }}
        >
          {number}
        </span>
        Step {number}: {title}
      </h3>
      <div className="pl-10 space-y-3 text-base leading-relaxed" style={{ color: "var(--color-text)" }}>
        {children}
      </div>
    </div>
  );
}

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

function CrossElectionTable({
  rows,
}: {
  rows: { cycle: string; r: number }[];
}) {
  // Normalize to [0, 1] range for bar width — max r in set determines 100%
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

function Divider() {
  return (
    <hr
      className="my-10"
      style={{ border: "none", borderTop: "1px solid var(--color-border)" }}
    />
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
      { rootMargin: "-20% 0px -60% 0px", threshold: 0 },
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

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 pb-20 flex gap-12">
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

        {/* Page hero */}
        <header className="mb-10">
          <p
            className="text-xs font-semibold uppercase tracking-widest mb-3"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-sans)" }}
          >
            WetherVane
          </p>
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

        <Divider />

        {/* 1 — The Key Insight */}
        <ExpandableSection id="key-insight" title="The Key Insight">
          <p>
            Most forecasting models treat counties as independent: average the
            polls, adjust for house effects, output a number. WetherVane starts
            from a different question:{" "}
            <em>what is the underlying structure that makes places move together?</em>
          </p>
          <p>
            Thousands of places share hidden behavioral patterns. A rural
            evangelical county in Georgia moves with rural evangelical counties
            in Iowa — not because anyone coordinates, but because the same
            forces act on similar communities. Discovering that structure —
            not just reading the surface results — is what makes prediction
            defensible.
          </p>
          <p>
            The model is built for readers who want to understand electoral
            dynamics, not just consume a top-line forecast. If you read
            FiveThirtyEight for the methodology write-ups, this is for you.
          </p>
        </ExpandableSection>

        <Divider />

        {/* 2 — How It Works */}
        <ExpandableSection id="how-it-works" title="How It Works">
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
        </ExpandableSection>

        <Divider />

        {/* 3 — Model Performance */}
        <ExpandableSection id="performance" title="Model Performance">
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
        </ExpandableSection>

        <Divider />

        {/* 4 — Historical Accuracy */}
        <ExpandableSection id="historical-accuracy" title="Historical Accuracy">
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
        </ExpandableSection>

        <Divider />

        {/* 5 — What Makes This Different */}
        <ExpandableSection id="differentiation" title="What Makes This Different">
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
        </ExpandableSection>

        <Divider />

        {/* 6 — Data Sources */}
        <ExpandableSection id="data-sources" title="Data Sources">
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
        </ExpandableSection>

        <Divider />

        {/* 7 — Current Status */}
        <ExpandableSection id="status" title="Current Status">
          <p>
            WetherVane is in active development, targeting the{" "}
            <strong>2026 midterm elections</strong>. The model currently covers
            all 50 states and DC, tracking 18 competitive races.
          </p>
          <p>
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
              { label: "Races tracked", value: "18" },
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
        </ExpandableSection>

        <Divider />

        {/* 8 — Credits */}
        <ExpandableSection id="credits" title="Credits">
          <p>
            Built by <strong>Hayden Haines</strong>.
          </p>
          <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
            Methodology inspired by The Economist&apos;s 2020 presidential model
            (Heidemanns, Gelman &amp; Morris). Type-covariance architecture
            adapted to shift-based community discovery.
          </p>
          <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
            Election return data from MIT MEDSL and Algara &amp; Amlani (Harvard
            Dataverse). All other data sources are listed above.
          </p>
        </ExpandableSection>

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
