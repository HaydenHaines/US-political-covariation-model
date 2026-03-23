export function MethodologyView() {
  return (
    <div style={{ padding: "16px" }}>

      {/* Title */}
      <h1 style={{
        fontFamily: "var(--font-serif)",
        fontSize: "22px",
        fontWeight: "700",
        margin: "0 0 4px",
      }}>
        Methodology
      </h1>
      <p style={{
        fontSize: "13px",
        color: "var(--color-text-muted)",
        margin: "0 0 20px",
        lineHeight: "1.5",
      }}>
        How Bedrock discovers electoral structure and produces forecasts.
      </p>

      <Divider />

      {/* Section 1: What is Bedrock? */}
      <Section title="What is Bedrock?">
        <p>
          Bedrock is a structural model of electoral behavior. Rather than treating
          polls as the whole story, it discovers hidden patterns in how communities
          shift politically across elections — and uses that structure to propagate
          new information across geography.
        </p>
        <p>
          Most forecasting models are poll aggregators: they average polls, adjust
          for house effects, and output a number. Bedrock starts from a different
          question: <em>what is the underlying structure that makes places move
          together?</em> Understanding that structure — not just the surface
          results — is what makes prediction defensible.
        </p>
        <p>
          The target audience is anyone who wants to understand electoral dynamics,
          not just consume a top-line forecast. If you read FiveThirtyEight for the
          methodology write-ups, this is for you.
        </p>
      </Section>

      <Divider />

      {/* Section 2: How It Works */}
      <Section title="How It Works">
        <Step number={1} title="Discover Types">
          <p>
            The model begins by computing how every county shifted politically
            across each pair of elections from 2008 to 2024. These shift vectors
            capture the direction and magnitude of change — did this county swing
            toward Democrats or Republicans, and by how much?
          </p>
          <p>
            KMeans clustering (J=43) groups counties with similar shift patterns
            into <strong>electoral types</strong>. Presidential shifts are
            weighted <strong>2.5x</strong> because they carry cross-state signal.
            Governor and Senate shifts are state-centered first (subtracting the
            statewide swing) so that clustering captures within-state variation,
            not just red-state/blue-state differences.
          </p>
        </Step>

        <Step number={2} title="Soft Membership">
          <p>
            No county belongs to just one type. Each county has partial membership
            in multiple types, computed via temperature-scaled inverse distance in
            shift space. A suburban Atlanta county might be 40% "College-Educated
            Suburban" and 30% "Black Belt &amp; Diverse."
          </p>
          <p>
            This soft membership (temperature T=10) reduces calibration error
            by ~37% compared to hard assignment, because real places are
            mixtures.
          </p>
        </Step>

        <Step number={3} title="Covariance Structure">
          <p>
            Types that share demographic profiles tend to move together
            politically. The model estimates a covariance matrix capturing how
            much each pair of types co-moves, using demographic profile
            correlation with Ledoit-Wolf regularization.
          </p>
          <p>
            This is validated against observed electoral comovement — the
            demographic construction must agree with what actually happened, or
            the model falls back to direct observation.
          </p>
        </Step>

        <Step number={4} title="Poll Propagation">
          <p>
            When a new poll comes in, the model does not just update the polled
            state. It propagates information to every county through the type
            covariance structure. A Florida poll shifts Georgia predictions
            too, because both states contain counties that share type
            memberships.
          </p>
          <p>
            The update mechanism is a Bayesian Gaussian (Kalman filter)
            update — exact and closed-form, no simulation needed.
          </p>
        </Step>

        <Step number={5} title="County Predictions">
          <p>
            Each county's prediction starts from its own historical baseline
            (its actual results in prior elections), then adjusts based on how
            much its types shifted according to polls. The adjustment respects
            each county's unique type composition — two counties in the same
            state can move in opposite directions if their type profiles differ.
          </p>
        </Step>
      </Section>

      <Divider />

      {/* Section 3: What Makes This Different */}
      <Section title="What Makes This Different">
        <BulletList items={[
          <>
            <strong>Structure from behavior, not demographics.</strong>{" "}
            Types are discovered from how places shift electorally. Demographics
            describe the types after discovery — they do not define them. This
            avoids baking in assumptions about which voters matter.
          </>,
          <>
            <strong>Cross-state information sharing.</strong>{" "}
            Because types cross state lines, a poll in one state informs
            predictions in another. Most models treat states as independent.
          </>,
          <>
            <strong>Full uncertainty quantification.</strong>{" "}
            Every prediction comes with 90% credible intervals, not just a
            point estimate.
          </>,
          <>
            <strong>Transparent and interpretable.</strong>{" "}
            Every prediction traces back to specific types, their shift
            patterns, and the polls that influenced them. This is not a black
            box — it is a structural model with components you can inspect.
          </>,
        ]} />
      </Section>

      <Divider />

      {/* Section 4: Current Status */}
      <Section title="Current Status">
        <p>
          Bedrock is in active development, targeting the 2026 midterm elections.
          The current pilot covers three states.
        </p>
        <MetricGrid metrics={[
          { label: "Pilot region", value: "FL, GA, AL" },
          { label: "Counties", value: "293" },
          { label: "Fine types", value: "43" },
          { label: "Super-types", value: "5" },
          { label: "County holdout r", value: "0.818" },
          { label: "Calibration MAE", value: "6.1 pp" },
          { label: "County RMSE", value: "2.67 pp" },
          { label: "KMeans stability", value: "r = 0.919 \u00B1 0.004" },
        ]} />
        <p style={{ marginTop: "12px" }}>
          <strong>Planned:</strong> National expansion (all 50 states + DC),
          Senate and Governor race-specific models.
        </p>
      </Section>

      <Divider />

      {/* Section 5: Data Sources */}
      <Section title="Data Sources">
        <p>
          Bedrock uses exclusively free, public data. No proprietary datasets or
          paid subscriptions.
        </p>
        <SourceList sources={[
          { name: "Election returns", source: "MIT Election Data & Science Lab (MEDSL)" },
          { name: "Demographics", source: "U.S. Census Bureau (Decennial + ACS)" },
          { name: "Religious congregations", source: "Association of Religion Data Archives (ARDA)" },
          { name: "Polling data", source: "FiveThirtyEight archives" },
          { name: "Governor returns", source: "Algara & Amlani (Harvard Dataverse)" },
        ]} />
      </Section>

      <Divider />

      {/* Section 6: Credits */}
      <Section title="Credits">
        <p>Built by Hayden Haines.</p>
        <p style={{ fontSize: "13px", color: "var(--color-text-muted)" }}>
          Methodology inspired by The Economist's 2020 presidential model
          (Heidemanns, Gelman, Morris).
        </p>
      </Section>

      {/* Bottom padding */}
      <div style={{ height: "32px" }} />
    </div>
  );
}


/* ── Sub-components ─────────────────────────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: "8px" }}>
      <h2 style={{
        fontFamily: "var(--font-serif)",
        fontSize: "17px",
        fontWeight: "700",
        margin: "0 0 10px",
      }}>
        {title}
      </h2>
      <div style={{ fontSize: "14px", lineHeight: "1.6", color: "var(--color-text)" }}>
        {children}
      </div>
    </div>
  );
}

function Step({ number, title, children }: { number: number; title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: "16px" }}>
      <h3 style={{
        fontFamily: "var(--font-serif)",
        fontSize: "15px",
        fontWeight: "600",
        margin: "0 0 6px",
        color: "var(--color-text)",
      }}>
        <span style={{
          display: "inline-block",
          width: "22px",
          height: "22px",
          lineHeight: "22px",
          textAlign: "center",
          borderRadius: "50%",
          background: "var(--color-text)",
          color: "var(--color-surface)",
          fontSize: "12px",
          fontWeight: "700",
          marginRight: "8px",
          verticalAlign: "middle",
          fontFamily: "var(--font-sans)",
        }}>
          {number}
        </span>
        {title}
      </h3>
      <div style={{
        fontSize: "14px",
        lineHeight: "1.6",
        color: "var(--color-text)",
        paddingLeft: "30px",
      }}>
        {children}
      </div>
    </div>
  );
}

function Divider() {
  return (
    <hr style={{
      border: "none",
      borderTop: "1px solid var(--color-border)",
      margin: "16px 0",
    }} />
  );
}

function BulletList({ items }: { items: React.ReactNode[] }) {
  return (
    <ul style={{
      margin: "8px 0",
      paddingLeft: "18px",
      listStyleType: "disc",
    }}>
      {items.map((item, i) => (
        <li key={i} style={{ marginBottom: "10px", lineHeight: "1.6" }}>
          {item}
        </li>
      ))}
    </ul>
  );
}

function MetricGrid({ metrics }: { metrics: { label: string; value: string }[] }) {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: "8px",
      marginTop: "8px",
    }}>
      {metrics.map((m) => (
        <div key={m.label} style={{
          padding: "8px 10px",
          border: "1px solid var(--color-border)",
          borderRadius: "4px",
        }}>
          <div style={{
            fontSize: "11px",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            color: "var(--color-text-muted)",
            marginBottom: "2px",
          }}>
            {m.label}
          </div>
          <div style={{
            fontFamily: "var(--font-serif)",
            fontSize: "16px",
            fontWeight: "700",
          }}>
            {m.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function SourceList({ sources }: { sources: { name: string; source: string }[] }) {
  return (
    <div style={{ marginTop: "8px" }}>
      {sources.map((s) => (
        <div key={s.name} style={{
          padding: "6px 0",
          borderBottom: "1px solid var(--color-border)",
          display: "flex",
          justifyContent: "space-between",
          gap: "8px",
        }}>
          <span style={{ fontWeight: "600", fontSize: "13px" }}>{s.name}</span>
          <span style={{ fontSize: "13px", color: "var(--color-text-muted)", textAlign: "right" }}>
            {s.source}
          </span>
        </div>
      ))}
    </div>
  );
}
