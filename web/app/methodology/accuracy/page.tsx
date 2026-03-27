import type { Metadata } from "next";
import Link from "next/link";

// ── Metadata ──────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "Model Accuracy | WetherVane",
  description:
    "Backtesting results for the WetherVane electoral model: leave-one-out cross-validation across four presidential cycles, method progression from baseline to ensemble, and what r=0.711 means in practice.",
  openGraph: {
    title: "Model Accuracy | WetherVane",
    description:
      "Backtesting results for the WetherVane electoral model: LOO r=0.711 across 3,154 counties, validated across four presidential cycles.",
    type: "article",
    siteName: "WetherVane",
    images: [
      {
        url: "/methodology/accuracy/opengraph-image",
        width: 1200,
        height: 630,
        alt: "WetherVane Model Accuracy — LOO r=0.711 across 3,154 counties",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Model Accuracy | WetherVane",
    description:
      "LOO r=0.711 across 3,154 counties. Four-cycle cross-election validation. See how the WetherVane model performs out-of-sample.",
  },
};

// ── Sub-components ────────────────────────────────────────────────────────

function Section({
  title,
  id,
  children,
}: {
  title: string;
  id?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} style={{ marginBottom: "40px" }}>
      <h2
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "26px",
          fontWeight: "700",
          margin: "0 0 16px",
          lineHeight: "1.25",
          color: "var(--color-text)",
        }}
      >
        {title}
      </h2>
      <div
        style={{
          fontSize: "16px",
          lineHeight: "1.75",
          color: "var(--color-text)",
        }}
      >
        {children}
      </div>
    </section>
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

function MetricCard({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div
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
        {label}
      </div>
      <div
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "22px",
          fontWeight: "700",
          lineHeight: "1.1",
        }}
      >
        {value}
      </div>
      {note && (
        <div
          style={{
            fontSize: "12px",
            color: "var(--color-text-muted)",
            marginTop: "4px",
          }}
        >
          {note}
        </div>
      )}
    </div>
  );
}

// ── Bar chart components ──────────────────────────────────────────────────

const MAX_BAR_R = 0.75; // scale bars relative to this ceiling

function CycleBar({
  cycle,
  label,
  loo_r,
  isBest,
}: {
  cycle: string;
  label: string;
  loo_r: number;
  isBest?: boolean;
}) {
  const pct = (loo_r / MAX_BAR_R) * 100;
  return (
    <div style={{ marginBottom: "16px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: "6px",
          gap: "8px",
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontWeight: "600", fontSize: "15px" }}>{cycle}</span>
        <span style={{ fontSize: "14px", color: "var(--color-text-muted)" }}>
          {label}
        </span>
      </div>
      <div
        style={{
          height: "28px",
          background: "var(--color-border)",
          borderRadius: "4px",
          overflow: "hidden",
          position: "relative",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: isBest ? "#2166ac" : "#888",
            borderRadius: "4px",
            transition: "width 0.3s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            paddingRight: "8px",
          }}
        >
          <span
            style={{
              color: "white",
              fontSize: "13px",
              fontWeight: "700",
              fontFamily: "var(--font-sans)",
              whiteSpace: "nowrap",
            }}
          >
            r = {loo_r.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}

function MethodBar({
  method,
  loo_r,
  isBest,
}: {
  method: string;
  loo_r: number;
  isBest?: boolean;
}) {
  const pct = (loo_r / MAX_BAR_R) * 100;
  return (
    <div style={{ marginBottom: "14px" }}>
      <div
        style={{
          fontSize: "14px",
          fontWeight: isBest ? "700" : "400",
          marginBottom: "5px",
          color: isBest ? "var(--color-text)" : "var(--color-text-muted)",
        }}
      >
        {method}
      </div>
      <div
        style={{
          height: "24px",
          background: "var(--color-border)",
          borderRadius: "4px",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: isBest ? "#2166ac" : "#aaa",
            borderRadius: "4px",
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            paddingRight: "8px",
          }}
        >
          <span
            style={{
              color: "white",
              fontSize: "12px",
              fontWeight: "700",
              fontFamily: "var(--font-sans)",
              whiteSpace: "nowrap",
            }}
          >
            {loo_r.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Page Component ────────────────────────────────────────────────────────

export default function AccuracyPage() {
  const crossElection = [
    { cycle: "2008→2012", label: "Obama→Obama", loo_r: 0.45 },
    { cycle: "2012→2016", label: "Obama→Trump", loo_r: 0.64 },
    { cycle: "2016→2020", label: "Trump→Biden", loo_r: 0.42 },
    { cycle: "2020→2024", label: "Biden→Trump", loo_r: 0.40 },
  ];

  const methodComparison = [
    { method: "Type-mean baseline", loo_r: 0.448 },
    { method: "Ridge regression (type scores only)", loo_r: 0.533 },
    { method: "Ridge regression (all 160 features)", loo_r: 0.671 },
    { method: "Ridge + HGB ensemble (production)", loo_r: 0.711 },
  ];

  return (
    <article
      id="main-content"
      style={{
        maxWidth: "720px",
        margin: "0 auto",
        padding: "40px 24px 80px",
      }}
    >
      {/* Back navigation */}
      <nav style={{ marginBottom: "32px", display: "flex", gap: "20px" }}>
        <Link
          href="/methodology"
          style={{
            color: "var(--color-dem)",
            textDecoration: "none",
            fontSize: "14px",
            fontWeight: "600",
          }}
        >
          ← Methodology
        </Link>
      </nav>

      {/* Hero */}
      <div style={{ marginBottom: "48px" }}>
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
          Model Accuracy
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
          Predictions are only valuable if they can be verified. This page
          shows how WetherVane performs on elections it has never seen —
          the only test that matters.
        </p>
      </div>

      <Divider />

      {/* Overall Performance */}
      <Section title="Overall Performance" id="overall">
        <p>
          The headline metric is{" "}
          <strong>leave-one-out (LOO) r = 0.711</strong>. This is the
          correlation between predicted and actual county-level Democratic
          vote share shifts, measured on held-out counties that were
          excluded from their own type average before prediction.
        </p>
        <p>
          LOO is a stricter test than the standard holdout (r = 0.698)
          because it prevents any county from inflating the score by
          predicting itself. The ~0.013 gap between them is small, which
          means type generalizations are stable.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: "12px",
            marginTop: "20px",
            marginBottom: "8px",
          }}
        >
          <MetricCard label="LOO r (ensemble)" value="0.711" note="Ridge + HGB, N=3,106" />
          <MetricCard label="Standard holdout r" value="0.698" note="Inflated ~0.013 by self-prediction" />
          <MetricCard label="RMSE" value="7.3 pp" note="Percentage-point error" />
          <MetricCard label="Covariance val r" value="0.915" note="Observed Ledoit-Wolf" />
          <MetricCard label="Type coherence" value="0.783" note="Within-type consistency" />
          <MetricCard label="Counties covered" value="3,154" note="All 50 states + DC" />
        </div>
      </Section>

      <Divider />

      {/* Cross-Election Validation */}
      <Section title="Cross-Election Validation" id="cross-election">
        <p>
          The more demanding test is cross-election validation: train the
          type structure on elections up to year N, then predict shifts
          in year N+4. This measures whether the discovered community
          types are durable structural features of American politics —
          or just noise in one election cycle.
        </p>
        <p>
          Across four presidential cycles, mean LOO r ={" "}
          <strong>0.476 ± 0.10</strong>. The variance is real and
          interpretable: not all elections test the same thing.
        </p>

        <div style={{ marginTop: "24px" }}>
          {crossElection.map((e) => (
            <CycleBar
              key={e.cycle}
              cycle={e.cycle}
              label={e.label}
              loo_r={e.loo_r}
              isBest={e.loo_r === Math.max(...crossElection.map((x) => x.loo_r))}
            />
          ))}
        </div>

        <div
          style={{
            marginTop: "20px",
            padding: "16px 20px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "6px",
            fontSize: "14px",
            lineHeight: "1.65",
          }}
        >
          <strong>Why 2024 was hardest (r = 0.40).</strong> The
          Biden-to-Trump cycle saw unusual cross-type movement —
          particularly among Hispanic communities, which broke sharply
          from their historical type patterns. When entire demographic
          groups shift in ways that cut across the type structure, the
          model&apos;s ability to predict from prior shifts degrades.
          This is a known limitation: the model captures structural
          patterns, not realignments in progress.
        </div>

        <div
          style={{
            marginTop: "12px",
            padding: "16px 20px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "6px",
            fontSize: "14px",
            lineHeight: "1.65",
          }}
        >
          <strong>Why 2012→2016 was most predictable (r = 0.64).</strong>{" "}
          Trump&apos;s initial surge in 2016 followed existing type fault
          lines closely. Rural working-class counties that were already
          trending Republican continued their trajectory; college-educated
          suburban counties that were already competitive moved further
          toward Democrats. The type structure discovered from 2008–2012
          data captured exactly these patterns.
        </div>
      </Section>

      <Divider />

      {/* Method Progression */}
      <Section title="How the Model Improved" id="method-progression">
        <p>
          The production model is not a single algorithm — it is the result
          of systematic improvement from a simple baseline. Each step
          added information while maintaining leave-one-out honesty.
        </p>

        <div style={{ marginTop: "24px" }}>
          {methodComparison.map((m) => (
            <MethodBar
              key={m.method}
              method={m.method}
              loo_r={m.loo_r}
              isBest={m.loo_r === Math.max(...methodComparison.map((x) => x.loo_r))}
            />
          ))}
        </div>

        <div style={{ marginTop: "20px", fontSize: "14px", color: "var(--color-text-muted)" }}>
          <p style={{ margin: "0 0 8px" }}>
            <strong style={{ color: "var(--color-text)" }}>Type-mean baseline:</strong> Predict each
            county from its type&apos;s average shift, excluding the county itself (LOO). This is the
            structural model alone — no demographics, no external data.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            <strong style={{ color: "var(--color-text)" }}>Ridge (scores only):</strong> Use all 100
            type membership scores as features in a Ridge regression. Captures nonlinear type
            interactions.
          </p>
          <p style={{ margin: "0 0 8px" }}>
            <strong style={{ color: "var(--color-text)" }}>Ridge (all features):</strong> Add 59
            features from 8 independent sources: ACS demographics, religious congregations, BLS
            industry composition, County Health Rankings, IRS migration flows, Facebook Social
            Connectedness Index, urbanicity, and broadband access.
          </p>
          <p style={{ margin: "0" }}>
            <strong style={{ color: "var(--color-text)" }}>Ridge + HGB ensemble:</strong> Combine
            Ridge predictions with a Histogram Gradient Boosted tree trained on the same 160
            features. The ensemble captures patterns that neither model finds alone.
          </p>
        </div>
      </Section>

      <Divider />

      {/* What This Means */}
      <Section title="What This Means" id="interpretation">
        <p>
          An LOO r of 0.711 means the model explains roughly{" "}
          <strong>50% of the variance</strong> in county-level partisan
          shifts (r² ≈ 0.505). That sounds modest, but consider what
          is being predicted: the direction and magnitude of how each of
          3,154 counties shifts relative to the prior election, using
          only information available <em>before</em> the election.
        </p>
        <p>
          The other 50% of variance is genuinely unpredictable from
          structural features — candidate effects, late-breaking news,
          local mobilization, and pure noise. No structural model can
          capture these, and one that claimed to would be overfitting.
          The model is designed to capture the part that <em>is</em>{" "}
          predictable: the structural landscape of which communities
          tend to move together.
        </p>
        <p>
          The model performs best on counties whose type membership is
          concentrated — places that are clearly one type tend to behave
          predictably. It performs worst on counties that are at the
          boundary between types, and on cycles where entire demographic
          groups cross type boundaries (like Hispanic communities in
          2024). These limitations are documented, not hidden.
        </p>
        <p style={{ color: "var(--color-text-muted)", fontSize: "14px" }}>
          The standard holdout r (0.698) is slightly higher than LOO (0.711) is
          lower because the standard metric allows each county to predict its
          own type mean. LOO excludes the county being predicted, which is the
          correct evaluation. Both are reported for full transparency.
        </p>
      </Section>

      <Divider />

      {/* Footer nav */}
      <div
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
          href="/methodology"
          style={{
            color: "var(--color-dem)",
            textDecoration: "none",
            fontSize: "14px",
            fontWeight: "600",
          }}
        >
          ← Full methodology
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
      </div>
    </article>
  );
}
