"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { ThemeToggle } from "@/components/ThemeToggle";

// ── Table of Contents data ────────────────────────────────────────────────

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

// ── Expandable Section ───────────────────────────────────────────────────

function ExpandableSection({
  title,
  id,
  defaultExpanded = true,
  children,
}: {
  title: string;
  id: string;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <section id={id} style={{ marginBottom: "40px" }}>
      <button
        className="expandable-header"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls={`${id}-body`}
        type="button"
      >
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "26px",
            fontWeight: "700",
            margin: "0",
            lineHeight: "1.25",
            color: "var(--color-text)",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          {title}
          <a
            href={`#${id}`}
            className="section-anchor"
            aria-label={`Link to ${title} section`}
            onClick={(e) => e.stopPropagation()}
          >
            #
          </a>
        </h2>
        <svg
          className={`expand-icon ${expanded ? "expanded" : ""}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      <div
        id={`${id}-body`}
        className={`expandable-body ${expanded ? "expanded" : "collapsed"}`}
        role="region"
        aria-labelledby={id}
      >
        <div
          style={{
            fontSize: "16px",
            lineHeight: "1.75",
            color: "var(--color-text)",
            paddingTop: "16px",
          }}
        >
          {children}
        </div>
      </div>
    </section>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────

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
    <div style={{ marginBottom: "28px" }}>
      <h3
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "18px",
          fontWeight: "600",
          margin: "0 0 10px",
          color: "var(--color-text)",
          display: "flex",
          alignItems: "center",
          gap: "12px",
        }}
      >
        <span
          aria-hidden="true"
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            width: "28px",
            height: "28px",
            borderRadius: "50%",
            background: "var(--color-text)",
            color: "var(--color-surface)",
            fontSize: "13px",
            fontWeight: "700",
            fontFamily: "var(--font-sans)",
          }}
        >
          {number}
        </span>
        <span>Step {number}: {title}</span>
      </h3>
      <div
        style={{
          fontSize: "16px",
          lineHeight: "1.75",
          color: "var(--color-text)",
          paddingLeft: "40px",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function Divider() {
  return (
    <hr
      style={{
        border: "none",
        borderTop: "1px solid var(--color-border)",
        margin: "40px 0",
      }}
    />
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
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "12px",
        marginTop: "16px",
        marginBottom: "8px",
      }}
    >
      {metrics.map((m) => (
        <div
          key={m.label}
          role="listitem"
          style={{
            padding: "14px 16px",
            border: "1px solid var(--color-border)",
            borderRadius: "6px",
            background: "var(--color-surface)",
          }}
        >
          <div
            style={{
              fontSize: "11px",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: "var(--color-text-muted)",
              marginBottom: "4px",
            }}
          >
            {m.label}
          </div>
          <div
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: "22px",
              fontWeight: "700",
              lineHeight: "1.1",
            }}
          >
            {m.value}
          </div>
          {m.note && (
            <div
              style={{
                fontSize: "12px",
                color: "var(--color-text-muted)",
                marginTop: "4px",
              }}
            >
              {m.note}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function SourceRow({ name, source }: { name: string; source: string }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        gap: "16px",
        padding: "10px 0",
        borderBottom: "1px solid var(--color-border)",
        flexWrap: "wrap",
      }}
    >
      <span style={{ fontWeight: "600", fontSize: "15px", flexShrink: 0 }}>
        {name}
      </span>
      <span
        style={{
          fontSize: "14px",
          color: "var(--color-text-muted)",
          textAlign: "right",
        }}
      >
        {source}
      </span>
    </div>
  );
}

// ── Table of Contents sidebar ────────────────────────────────────────────

function TableOfContents({ activeId }: { activeId: string }) {
  return (
    <nav className="methodology-toc" aria-label="Table of contents">
      <div style={{
        fontSize: "11px",
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        color: "var(--color-text-muted)",
        fontWeight: "700",
        padding: "0 12px 8px",
      }}>
        On this page
      </div>
      <ul className="methodology-toc-list">
        {TOC_SECTIONS.map((section) => (
          <li key={section.id} className="methodology-toc-item">
            <a
              href={`#${section.id}`}
              className={`methodology-toc-link ${activeId === section.id ? "active" : ""}`}
            >
              {section.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function MethodologyContent() {
  const [activeSection, setActiveSection] = useState("key-insight");
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Track which section is in view via IntersectionObserver
  const setupObserver = useCallback(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        // Find the topmost visible section
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

        if (visible.length > 0) {
          setActiveSection(visible[0].target.id);
        }
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: 0 },
    );

    for (const section of TOC_SECTIONS) {
      const el = document.getElementById(section.id);
      if (el) {
        observerRef.current.observe(el);
      }
    }

    return () => observerRef.current?.disconnect();
  }, []);

  useEffect(() => {
    const cleanup = setupObserver();
    return cleanup;
  }, [setupObserver]);

  return (
    <div className="methodology-layout">
      <TableOfContents activeId={activeSection} />

      <div className="methodology-content">
        {/* Header bar with back link and theme toggle */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "32px",
        }}>
          <nav aria-label="Breadcrumb">
            <Link
              href="/forecast"
              style={{
                color: "var(--color-dem)",
                textDecoration: "none",
                fontSize: "14px",
                fontWeight: "600",
              }}
            >
              ← Back to map
            </Link>
          </nav>
          <ThemeToggle />
        </div>

        {/* Hero */}
        <header style={{ marginBottom: "48px" }}>
          <p
            style={{
              fontFamily: "var(--font-sans)",
              fontSize: "13px",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--color-text-muted)",
              margin: "0 0 12px",
            }}
          >
            WetherVane
          </p>
          <h1
            style={{
              fontFamily: "var(--font-serif)",
              fontSize: "42px",
              fontWeight: "700",
              margin: "0 0 20px",
              lineHeight: "1.15",
            }}
          >
            How WetherVane Works
          </h1>
          <p
            style={{
              fontSize: "18px",
              lineHeight: "1.75",
              color: "var(--color-text)",
              margin: "0",
              borderLeft: "3px solid var(--color-border)",
              paddingLeft: "20px",
            }}
          >
            WetherVane discovers communities of voters who move together
            politically — then uses that structure to propagate new information
            across geography. A poll in one state updates predictions in every
            state that shares those communities.
          </p>
        </header>

        <Divider />

        {/* The Key Insight */}
        <ExpandableSection title="The Key Insight" id="key-insight">
          <p>
            Most forecasting models treat counties as independent: average the
            polls, adjust for house effects, output a number. WetherVane starts
            from a different question: <em>what is the underlying structure that
            makes places move together?</em>
          </p>
          <p>
            Thousands of places share hidden behavioral patterns. A rural
            evangelical county in Georgia moves with rural evangelical counties in
            Iowa — not because anyone coordinates, but because the same forces act
            on similar communities. Discovering that structure — not just reading
            the surface results — is what makes prediction defensible.
          </p>
          <p>
            The model is built for readers who want to understand electoral
            dynamics, not just consume a top-line forecast. If you read
            FiveThirtyEight for the methodology write-ups, this is for you.
          </p>
        </ExpandableSection>

        <Divider />

        {/* How It Works */}
        <ExpandableSection title="How It Works" id="how-it-works">
          <Step number={1} title="Discover Types">
            <p>
              The model begins by computing how every county shifted politically
              across each pair of elections from 2008 to 2024. These shift vectors
              capture direction and magnitude — did this county swing toward
              Democrats or Republicans, and by how much?
            </p>
            <p>
              KMeans clustering (<strong>J=100</strong>) groups counties with
              similar shift patterns into <strong>electoral types</strong>.
              Presidential shifts are weighted <strong>8x</strong> because they
              carry cross-state signal (the same national forces act on similar
              communities everywhere). Governor and Senate shifts are{" "}
              <em>state-centered</em> first — subtracting the statewide swing —
              so that clustering captures within-state variation, not just
              red-state/blue-state geography.
            </p>
            <p>
              The result is 100 fine-grained electoral types and 5{" "}
              <strong>super-types</strong> (broad behavioral families), which
              form the colors of the stained glass map.
            </p>
          </Step>

          <Step number={2} title="Soft Membership">
            <p>
              No county belongs to just one type. Each county has partial
              membership in multiple types, computed via temperature-scaled
              inverse distance in shift space. A suburban Atlanta county might be
              40% &quot;College-Educated Suburban&quot; and 30% &quot;Black Belt &amp; Diverse.&quot;
            </p>
            <p>
              This soft membership (temperature <strong>T=10</strong>) reduces
              calibration error by ~37% compared to hard assignment, because real
              places are mixtures. The map color reflects dominant type, but
              predictions use the full membership vector.
            </p>
          </Step>

          <Step number={3} title="Covariance Structure">
            <p>
              Types that share electoral behavior tend to move together
              politically. The model estimates a 100x100 covariance matrix
              capturing how much each pair of types co-moves, using observed
              electoral correlation with Ledoit-Wolf regularization.
            </p>
            <p>
              This observed covariance (validation r = <strong>0.915</strong>)
              is the key structural object. It encodes which types are
              behaviorally coupled — and therefore how information should flow
              between them when a new poll arrives.
            </p>
          </Step>

          <Step number={4} title="Poll Propagation">
            <p>
              When a new poll comes in, the model does not just update the polled
              state. It propagates information to every county through the type
              covariance structure. A Florida Senate poll shifts Georgia
              predictions too, because both states contain counties that share
              type memberships.
            </p>
            <p>
              The update mechanism is a{" "}
              <strong>Bayesian Gaussian (Kalman filter) update</strong> — exact
              and closed-form, no simulation needed. Multiple polls stack as
              independent observations rather than collapsing to an average,
              preserving the information in each.
            </p>
          </Step>

          <Step number={5} title="County Predictions">
            <p>
              Each county&apos;s prediction starts from its historical baseline
              (actual results in prior elections), then adjusts based on how much
              its types shifted according to polls. The adjustment respects each
              county&apos;s unique type composition — two counties in the same
              state can move in opposite directions if their type profiles differ.
            </p>
            <p>
              The final ensemble uses a{" "}
              <strong>Ridge + Histogram Gradient Boosting</strong> combination
              with 160 features from 8 independent data sources, achieving a
              leave-one-out r of <strong>0.711</strong> — the honest
              out-of-sample accuracy measure.
            </p>
          </Step>
        </ExpandableSection>

        <Divider />

        {/* Model Performance */}
        <ExpandableSection title="Model Performance" id="performance">
          <p>
            All metrics are on the 2024 presidential election. LOO (leave-one-out)
            cross-validation excludes each county from its own type mean before
            predicting it — this is the honest generalization metric that cannot
            be inflated by self-prediction.
          </p>
          <MetricGrid
            metrics={[
              { label: "Counties", value: "3,154", note: "All 50 states + DC" },
              { label: "Fine types", value: "100", note: "KMeans clusters" },
              { label: "Super-types", value: "5", note: "Behavioral families" },
              {
                label: "Ensemble LOO r",
                value: "0.711",
                note: "Ridge + HGB, N=3,106",
              },
              { label: "County RMSE", value: "7.3 pp", note: "Percentage points" },
              {
                label: "Covariance val r",
                value: "0.915",
                note: "Ledoit-Wolf observed",
              },
              { label: "Type coherence", value: "0.783", note: "Within-type consistency" },
              {
                label: "Cross-election LOO r",
                value: "0.476 +/- 0.10",
                note: "Mean over 4 cycles",
              },
            ]}
          />
        </ExpandableSection>

        <Divider />

        {/* Historical Accuracy */}
        <ExpandableSection title="Historical Accuracy" id="historical-accuracy">
          <p>
            The model&apos;s honest test is leave-one-out cross-validation:
            predict each county while excluding it from its type average. The
            ensemble achieves <strong>LOO r=0.711</strong> across 3,106 matched
            counties, using 160 features from 8 independent data sources.
          </p>
          <p>
            Cross-election validation tests whether the type structure discovered
            from historical shifts generalizes to new cycles. Across four
            presidential elections (2012-2016, 2016-2020, 2020-2024, and one
            earlier cycle), mean LOO r = <strong>0.476 +/- 0.10</strong>.
          </p>
          <p>
            Not all cycles are equally predictable.{" "}
            <strong>2024 (r=0.40)</strong> was the hardest — the Harris-Trump
            dynamic produced unusual cross-type movement, particularly among
            Hispanic communities. The{" "}
            <strong>2012-to-2016 transition (r=0.64)</strong> was most predictable:
            Trump&apos;s initial surge followed existing type fault lines closely.
          </p>
          <p style={{ color: "var(--color-text-muted)", fontSize: "14px" }}>
            Note: The standard holdout r (0.698) is inflated by ~0.22 because
            counties predict their own type means. LOO is the correct metric for
            evaluating generalization. Both are reported for transparency.
          </p>
        </ExpandableSection>

        <Divider />

        {/* What Makes This Different */}
        <ExpandableSection title="What Makes This Different" id="differentiation">
          <ul
            style={{
              margin: "0",
              paddingLeft: "24px",
              listStyleType: "disc",
            }}
          >
            <li style={{ marginBottom: "14px" }}>
              <strong>Structure from behavior, not demographics.</strong> Types
              are discovered from how places shift electorally. Demographics
              describe the types after discovery — they do not define them. This
              avoids baking in assumptions about which demographic groups drive
              politics.
            </li>
            <li style={{ marginBottom: "14px" }}>
              <strong>Cross-state information sharing.</strong> Because types
              cross state lines, a poll in one state informs predictions in
              another. Most models treat states as independent. WetherVane
              treats the country as one connected landscape.
            </li>
            <li style={{ marginBottom: "14px" }}>
              <strong>Full uncertainty quantification.</strong> Every prediction
              comes with 90% credible intervals, not just a point estimate. The
              intervals widen where the model has less data and tighten where type
              signals are strong.
            </li>
            <li style={{ marginBottom: "14px" }}>
              <strong>Transparent and interpretable.</strong> Every prediction
              traces back to specific types, their shift patterns, and the polls
              that influenced them. This is not a black box — it is a structural
              model with components you can inspect on{" "}
              <Link
                href="/forecast"
                style={{ color: "var(--color-dem)", textDecoration: "none" }}
              >
                the map
              </Link>
              .
            </li>
            <li style={{ marginBottom: "14px" }}>
              <strong>Free data only.</strong> No proprietary datasets, no paid
              subscriptions. Every source listed below is publicly available.
              This makes the model fully reproducible.
            </li>
          </ul>
        </ExpandableSection>

        <Divider />

        {/* Data Sources */}
        <ExpandableSection title="Data Sources" id="data-sources">
          <p>
            WetherVane uses exclusively free, public data. No proprietary datasets
            or paid subscriptions.
          </p>
          <div style={{ marginTop: "8px" }} role="list" aria-label="Data sources">
            <SourceRow
              name="Election returns"
              source="MIT Election Data & Science Lab (MEDSL)"
            />
            <SourceRow
              name="Demographics"
              source="US Census Bureau — Decennial 2000/2010/2020 + ACS 5-year"
            />
            <SourceRow
              name="Religious congregations"
              source="ARDA — Religious Congregations & Membership Study (RCMS 2020)"
            />
            <SourceRow
              name="Industry composition"
              source="BLS Quarterly Census of Employment and Wages (QCEW)"
            />
            <SourceRow
              name="Health behaviors"
              source="County Health Rankings (Robert Wood Johnson Foundation)"
            />
            <SourceRow
              name="Migration flows"
              source="IRS Statistics of Income (SOI) — county-to-county migration"
            />
            <SourceRow
              name="Social connectivity"
              source="Facebook Social Connectedness Index (county-pair network)"
            />
            <SourceRow
              name="Broadband access"
              source="FCC / ACS — internet subscription at county level"
            />
            <SourceRow
              name="Polling data"
              source="FiveThirtyEight archives + Silver Bulletin pollster ratings"
            />
            <SourceRow
              name="Governor returns"
              source="Algara & Amlani (Harvard Dataverse) — 2002-2022 governor"
            />
          </div>
        </ExpandableSection>

        <Divider />

        {/* Current Status */}
        <ExpandableSection title="Current Status" id="status">
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
          <p>
            <strong>Planned improvements:</strong> BEA regional economic data,
            FEC donor density features, richer poll ingestion with crosstab
            disaggregation (crosstabs tell us which types were sampled — a poll
            oversampling college-educated voters should pull harder on types with
            high college-educated membership).
          </p>
        </ExpandableSection>

        <Divider />

        {/* Credits */}
        <ExpandableSection title="Credits" id="credits" defaultExpanded>
          <p>
            Built by{" "}
            <strong>Hayden Haines</strong>.
          </p>
          <p style={{ fontSize: "15px", color: "var(--color-text-muted)" }}>
            Methodology inspired by The Economist&apos;s 2020 presidential model
            (Heidemanns, Gelman &amp; Morris). Type-covariance architecture
            adapted to shift-based community discovery.
          </p>
          <p style={{ fontSize: "15px", color: "var(--color-text-muted)" }}>
            Election return data from MIT MEDSL and Algara &amp; Amlani (Harvard
            Dataverse). All other data sources are listed above.
          </p>
        </ExpandableSection>

        {/* Accuracy page link */}
        <div
          style={{
            padding: "20px 24px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "8px",
            marginBottom: "40px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          <div>
            <div
              style={{
                fontFamily: "var(--font-serif)",
                fontSize: "17px",
                fontWeight: "700",
                marginBottom: "4px",
              }}
            >
              Backtesting results
            </div>
            <div style={{ fontSize: "14px", color: "var(--color-text-muted)" }}>
              Four-cycle cross-election validation / LOO r = 0.711
            </div>
          </div>
          <Link
            href="/methodology/accuracy"
            style={{
              color: "var(--color-dem)",
              textDecoration: "none",
              fontSize: "14px",
              fontWeight: "600",
              whiteSpace: "nowrap",
            }}
          >
            View accuracy →
          </Link>
        </div>

        {/* Footer nav */}
        <nav
          aria-label="Page navigation"
          style={{
            paddingTop: "24px",
            borderTop: "1px solid var(--color-border)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          <Link
            href="/forecast"
            style={{
              color: "var(--color-dem)",
              textDecoration: "none",
              fontSize: "14px",
              fontWeight: "600",
            }}
          >
            ← Back to map
          </Link>
          <Link
            href="/forecast"
            style={{
              color: "var(--color-dem)",
              textDecoration: "none",
              fontSize: "14px",
              fontWeight: "600",
            }}
          >
            View 2026 race forecasts →
          </Link>
        </nav>
      </div>
    </div>
  );
}
