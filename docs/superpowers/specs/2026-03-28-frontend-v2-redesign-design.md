# Design Spec: WetherVane Frontend V2 — The Academic Dream of Political Data Visualization

**Date:** 2026-03-28
**Status:** APPROVED
**Scope:** Complete frontend redesign — clean foundation rebuild of all pages, components, styling, data layer, and interaction patterns.
**Approach:** Clean build on a feature branch. No incremental migration. Every piece built right from day one.

---

## Why This Exists

> **Research foundation:** This spec synthesizes findings from four parallel research streams conducted 2026-03-28: a Playwright visual/functional audit of every page and interaction, a survey of best-in-class political data visualization (538, The Economist, NYT Upshot, DDHQ, 270toWin, WaPo, Pudding.cool, Reuters Graphics, Silver Bulletin, ProPublica), a GitHub/npm evaluation of 40+ visualization libraries, and a full architectural exploration of the existing 21-component codebase.
>
> **Supporting documents:**
> - Playwright audit: `docs/frontend-audit-2026-03-28.md` (29 issues, 3 critical)
> - Data viz research: `research/political-dataviz-research.md` (14 principles, 32 patterns, anti-patterns)
> - Codebase architecture: 21 components, ~5,800 lines, deck.gl + Observable Plot + custom CSS

The current frontend is a functional MVP. The methodology page is publication-quality. The stained glass map is a genuine differentiator. But the forecast page shows "EVEN" for every race, type names contradict their data, navigation is broken between pages, and the styling system (inline React styles + globals.css) cannot scale.

This redesign builds the frontend WetherVane deserves — one that treats political data visualization as a craft, not an afterthought.

---

## Governing Principles

These principles are derived directly from the research and must guide every implementation decision. When in doubt, refer back to these.

### P1: Uncertainty Is the Story, Not a Footnote
> **Research basis:** FiveThirtyEight's graduated confidence shading, The Economist's quantile dotplots, NYT Needle's jittering motion. See `research/political-dataviz-research.md` §1 Principle 1.

Never show a single number when you can show a distribution. Every prediction must include a confidence interval or probability range. Quantile dotplots are the primary uncertainty visualization.

### P2: Progressive Disclosure Over Information Overload
> **Research basis:** 270toWin's immediate clarity, all top sites layer from headline → state → county → tract. See research §1 Principle 3.

Start with the simplest possible view (who's ahead?) and let users drill down. Each layer answers a progressively more specific question.

### P3: The "One Big Number" Anchor
> **Research basis:** FiveThirtyEight, Silver Bulletin, and The Economist all lead with a single probability/margin in massive type. See research §1 Principle 13.

Every forecast page needs a single dominant visual that anchors the reader before exploration begins.

### P4: Context Beats Raw Numbers
> **Research basis:** NYT event-annotated polling charts, 538's "Path to 270" snake chart. See research §1 Principle 5.

Every number should answer "compared to what?" Show historical comparisons, event annotations, and relative positioning.

### P5: Configuration Over Hardcoding
> **Project-specific mandate from stakeholder.**

Any value that could change when the model retrains belongs in the API response or a config file — never in component JSX. A model retrain should require zero frontend code changes.

### P6: Mobile Is a Different Product
> **Research basis:** The Economist and FiveThirtyEight completely restructure on mobile. See research §1 Principle 8.

Maps become cards. Tables become lists. Charts simplify to headlines. This is not CSS media queries on the same layout.

### P7: Speed Creates Trust
> **Research basis:** DDHQ and 270toWin load instantly via pre-rendering. See research §1 Principle 9.

First meaningful paint under 1 second. Skeleton screens during data fetches. Progressive loading of geographic detail.

### P8: Show Your Work
> **Research basis:** 538 and The Economist publish methodology and source code. See research §1 Principle 10.

Methodology is a first-class section. Transparency builds trust.

### P9: Data Provenance on Every Display
> **Research basis:** Cook Political Report shows poll counts explicitly. Timestamps build confidence. See research §1 Principle 14.

Every data display must answer: When was this updated? How many sources? How confident are we?

### P10: No Tech Debt From Day One
> **Project-specific mandate from stakeholder.**

Every component has a loading state, an error state, and a skeleton. Every data field is config-driven. Every color is tokenized. Start clean, stay clean.

---

## Information Architecture

```
/                           → Landing/hero (not a redirect)
├── /forecast               → Forecast Hub (all race types)
│   ├── /forecast/senate     → Senate overview + balance bar
│   ├── /forecast/governor   → Governor overview
│   └── /forecast/[slug]     → Race detail page
├── /explore                → Interactive exploration hub
│   ├── /explore/types       → Type directory + scatter plot + comparison
│   ├── /explore/map         → Full-screen stained glass map
│   └── /explore/shifts      → Historical shift analysis
├── /county/[fips]          → County detail
├── /type/[id]              → Type detail
├── /methodology            → Scrollytelling methodology
│   └── /methodology/accuracy → Model validation deep-dive
├── /embed/[slug]           → Embeddable widgets
└── /about                  → About + attribution
```

### Navigation

**Global nav (persistent, sticky, minimal):** Logo (home link) | Forecast | Explore | Methodology

**Footer:** About | Methodology | GitHub | Embed Info | Attribution

**Cross-linking rules:**
- Every race card links to its race detail page
- Every type reference links to its type detail page
- Every county reference links to its county detail page
- "View on Map" links navigate to `/explore/map?focus=[id]` with the entity centered
- Breadcrumbs on all detail pages, truncating middle segments on mobile

---

## Tech Stack

### Additions

| Layer | Tool | Stars | Replaces | Why |
|-------|------|-------|----------|-----|
| UI components | shadcn/ui + Radix UI + Tailwind v4 | 110K | Custom CSS + inline styles | Copy-paste model, accessible primitives, dark mode built-in |
| Custom charts | visx (Airbnb) | 20.7K | — | Low-level D3 React primitives, ~15KB/module, tree-shakeable |
| Data tables | TanStack Table | 27.8K | Hand-rolled tables | Headless, sortable, filterable, virtualizable |
| Animation | Motion (Framer Motion) | 31.3K | CSS transitions only | React-native animation, layout animations, scroll-triggered |
| Data fetching | SWR | — | Raw fetch + useEffect | Stale-while-revalidate caching, automatic retry |
| Scrollytelling | react-scrollama | — | — | IntersectionObserver-based, no scroll hijacking |

### Retained

| Tool | Why |
|------|-----|
| deck.gl v9 | Already correct for 81K tract WebGL rendering. Keep. |
| Observable Plot | Secondary — quick analysis charts on methodology/accuracy pages |
| Next.js App Router | Keep. SSR + ISR + dynamic imports all needed. |
| React Context | Keep for map state. SWR handles data fetching state. |

### Removed

| Tool | Replaced By |
|------|-------------|
| All inline React styles | Tailwind utility classes |
| globals.css (custom) | Tailwind base + shadcn theme variables |
| Hand-rolled Card/Button/Tab/Tooltip | shadcn components |
| Raw fetch + useEffect for data | SWR hooks |

### shadcn Components to Install

Card, Button, Tabs, Select, Slider, Tooltip, Dialog, Sheet, Badge, Breadcrumb, Skeleton, DropdownMenu, Toggle, Separator, Alert, Combobox (for type search).

---

## Design System: "Dusty Ink" v2

### Color Tokens

> **Research basis:** Purple for tossup, not gray. Gray reads as "no data." Saturation gradients for margin. Always test in grayscale + deuteranopia. See research §1 Principle 7, §5 Anti-pattern 3.

**Partisan scale (Tailwind CSS variables):**
```
--forecast-safe-d: #2d4a6f       (deep blue)
--forecast-likely-d: #4b6d90     (medium blue)
--forecast-lean-d: #7e9ab5       (light blue)
--forecast-tossup: #8a6b8a       (purple — NOT gray)
--forecast-lean-r: #c4907a       (light rose)
--forecast-likely-r: #9e5e4e     (medium rose)
--forecast-safe-r: #6e3535       (deep rose)
```

**Surface tokens (light / dark):**
```
--color-bg: #f7f8fa / #1a1b1e
--color-surface: #ffffff / #25262b
--color-surface-elevated: #ffffff / #2c2d32
--color-border: #e0e0e0 / #3d3f44
--color-border-subtle: #f0eeeb / #2a2b2f
--color-text: #222222 / #e0e0e0
--color-text-muted: #666666 / #a0a0a0
```

**Super-type stained glass palette:** 8 colors for 8 super-types. Colors defined in `palette.ts` config. Super-type names and count come from the API `/super-types` endpoint — never hardcoded. When super-types grow from 6 to 8 to 10, add color entries to the palette config. Frontend renders whatever the API returns.

### Typography

- **Serif (headlines):** Georgia, "Times New Roman", serif
- **Sans (body, UI):** -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif
- **Mono (data figures):** "Tabular Nums" font-feature-settings for aligned columns in tables

### Spacing & Elevation

- **Spacing grid:** 4px base (Tailwind default)
- **Elevation:** 3-tier shadow system — `shadow-sm` (cards), `shadow-md` (popovers), `shadow-lg` (modals/sheets)

### Dark Mode

- CSS variables swap via `data-theme` attribute on `<html>`
- Inline script before React hydration prevents flash (keep existing pattern)
- `prefers-color-scheme` media query for system detection
- Theme toggle cycles: system → light → dark → system

---

## Page Designs

### Landing Page (`/`)

> **Research basis:** P3 — "One Big Number" anchor. See research §1 Principle 13. Every forecast site leads with a dominant metric.

**Structure (top to bottom):**

1. **Hero headline** — One dominant metric in massive type. Example: "Democrats favored to hold the Senate" with a large probability or margin number. Below: "Based on 108 polls, 130 electoral types, and the WetherVane covariance model."

2. **Mini stained-glass map** — Simplified national map (state-level fill). Not interactive. Visual anchor. Click navigates to `/forecast`.

3. **Race ticker** — Horizontal row of 6-8 most competitive races as compact cards. State abbreviation, margin, rating badge. Links to race detail pages.

4. **Three entry points** — Card grid:
   - "See the full forecast →" (`/forecast`)
   - "Explore electoral types →" (`/explore/types`)
   - "How the model works →" (`/methodology`)

5. **Freshness timestamp** — "Model last updated March 28, 2026 at 4:12 PM CT. 7 new polls incorporated."

6. **Footer**

**Performance:** SSR the hero number + mini map at build time. Hydrate ticker client-side via SWR.

---

### Forecast Hub (`/forecast`)

> **Research basis:** P2 — Progressive disclosure. P4 — Context beats raw numbers. See research §2 Pattern Library: Senate balance bar, race cards with sparklines.

**Hub page** with race-type tabs: Senate | Governor. Tab state lives in the URL (`/forecast/senate`, `/forecast/governor`) for shareability.

**Senate overview (`/forecast/senate`):**

1. **Senate balance bar** — Horizontal stacked bar. Each segment = one race, colored by rating, width proportional to confidence. Hover shows tooltip. Click navigates to race detail. Real margins, real colors, real links.

2. **"Key Races" grid** — 6-10 most competitive races as cards, sorted by competitiveness. Each card:
   - State name + office
   - **Big number: margin** (e.g., "D+3.2")
   - Rating badge with Dusty Ink color
   - Poll count + freshness ("4 polls, latest 3 days ago")
   - Mini sparkline if 3+ polls
   - **Card is a `<Link>` to `/forecast/[slug]`**

3. **"All Races" table** — TanStack Table. Sortable by state, margin, rating, poll count. Every row links to detail page.

4. **Map integration** — Left pane shows state-level forecast coloring. Click state → navigate to most prominent race detail.

**Governor overview (`/forecast/governor`):** Same structure. Count bar (D/R/Tossup governors) instead of balance bar.

---

### Race Detail (`/forecast/[slug]`)

> **Research basis:** P1 — Uncertainty is the story. P3 — One big number. The Economist's quantile dotplots, 538's graduated confidence. See research §4 Inspiration Gallery: Economist dotplots, 538 data-driven sentences.

1. **Hero section** — Candidate names flanking a large margin number. Rating badge. Confidence interval: "90% chance between D+1.2 and D+5.8."

2. **Quantile dotplot** — 100 dots arranged as a distribution, each representing one simulation outcome. Concrete frequency framing: "In 73 of 100 scenarios, the Democrat wins." Built with visx. **This is the signature visualization.**

3. **Poll tracker chart** — visx area chart. Polling average over time with confidence band (stepped gradient fan chart — research shows stepped reduces estimation error vs continuous). Event annotations where significant polls land.

4. **Electoral types breakdown** — Top 5-8 types in this state. Type name (descriptive, no partisan suffix), super-type color dot, predicted margin, population weight. Links to type detail pages. **This is WetherVane's unique analytical lens — no competitor has it.**

5. **County/tract table** — TanStack Table. Sortable predictions by geography. Margin, confidence interval, shift from last election.

6. **Poll table** — All polls for this race. Pollster, date, sample size, result, rating if available. Sorted by recency.

7. **Section weight sliders** — shadcn Slider components for model prior / state polls / national polls. Clear labels explaining what each slider does.

---

### Explore Section

#### Type Directory (`/explore/types`)

> **Research basis:** P2 — Progressive disclosure. P4 — Comparison is the heart of political analysis. See research §1 Principle 11.

1. **Type grid** — 130 types as compact cards, grouped by super-type with colored section headers. Each card: descriptive name (no Red/Blue suffix), super-type color dot, county count, lean badge, key demographics. Search bar filters in real-time. Jump links to super-type sections. Cards link to `/type/[id]`.

2. **Scatter plot** — Rebuilt with visx. Human-readable axis labels. Dots colored by super-type with named legend. Hover tooltip. Click → type detail Sheet slides in. Responsive resize.

3. **Comparison table** — TanStack Table + shadcn. Combobox type selector with search. Color-coded relative value cells. Shareable URL with query params encoding selected type IDs.

#### Full-Screen Map (`/explore/map`)

> **Research basis:** P7 — Speed creates trust. See research §2 Pattern: Level-of-detail rendering, vector tile streaming.

Dedicated page for the map without panel competition.

- Full-viewport deck.gl with stained glass choropleth
- Floating legend with actual super-type names
- State click → fly-to zoom + tract load
- Tract click → shadcn Sheet with demographics, type, predicted margin
- **Overlay toggle:** "Forecast" (dem share gradient) | "Types" (stained glass) | "Shifts" (new — red→blue shift arrows)
- Mobile: full viewport, bottom Sheet for legend/controls

#### Historical Shifts (`/explore/shifts`)

> **Research basis:** P4 — Context beats raw numbers. P11 — Comparison is the heart. See research §2 Pattern: Small multiples, bivariate scatter.

- **Small multiples** — One mini visx chart per super-type showing Dem margin shift across 2008→2024. Consistent scales. "Realignment at a glance."
- **Bivariate scatter** — Shift in one election pair vs another. Reveals which types moved together vs diverged.
- **Narrative callouts** — Annotated insights linking to type pages.

Lower priority than Forecast. Can ship as v2. IA slot reserved from day one.

---

### Detail Pages

#### Type Detail (`/type/[id]`)

> **Research basis:** P5 — Configuration over hardcoding. P4 — Context beats raw numbers.

1. **Hero** — Descriptive type name (NO partisan suffix), super-type color badge, county count, predicted lean as big number. Narrative description from API.

2. **Shift history chart** — visx line chart: Dem margin across every election cycle. Zero line. Area fill above/below zero. The type's "character arc."

3. **Demographics panel** — Config-driven generic renderer (see Configuration Architecture below). All labels, formats, and sort order from `display.ts`. New model features auto-display. Removed features auto-disappear.

4. **Member geography** — Mini stained glass map filtered to this type's counties/tracts. Collapsible member list grouped by state, each linking to `/county/[fips]`.

5. **Correlated types** — 3-4 most correlated types from covariance matrix (API data). Links to their type pages. Surfaces the model's structural insight.

#### County Detail (`/county/[fips]`)

1. **Hero** — County name, state, population, predicted margin as big number, type assignment linking to `/type/[id]`.

2. **Election history chart** — visx bar/lollipop: actual Dem margin every election 2008→2024. Predicted 2026 as distinct styled bar.

3. **Demographics** — Same config-driven renderer as type pages.

4. **"View on Map"** — Navigates to `/explore/map?focus=[fips]` with county centered and highlighted.

5. **Similar counties** — Same-type counties sorted by similarity. Links to county pages.

---

### Type Naming Rules

> **Research basis:** Playwright audit finding — "Black Blue" shows R+23, "White Red" applied to LA County (D+15, 48.7% Hispanic). Critical credibility issue.

**Rules (non-negotiable):**
- Type names are **descriptive only**: demographic character + geographic pattern
- Examples: "Hispanic Working-Class", "Affluent White Suburb", "Rural Evangelical Heartland"
- **No partisan suffix** — no "Red", "Blue", "Dem", "Rep" in the name
- Predicted lean is shown as a **separate data field**, never baked into the name
- Names come from the API. Frontend never constructs or modifies type names
- When model retrains and type composition changes, API generates new names. Frontend displays whatever it receives.

---

### Methodology — Scrollytelling (`/methodology`)

> **Research basis:** P4 — Scrollytelling for narrative, dashboards for exploration. P8 — Show your work. See research §1 Principle 4, §2 Pattern: Scrollytelling sections (WaPo, Reuters, Pudding).

Built with react-scrollama. Scroll-triggered transitions pair text with updating visualizations.

**8-step story arc:**

| Step | Text | Visualization |
|------|------|---------------|
| 1. "What if elections aren't random?" | Opening hook | Static stained glass map fades in |
| 2. "Discovering types from shifts" | KMeans on shift vectors | Scatter plot animates, centroids appear |
| 3. "The stained glass landscape" | Full national picture | Map renders, zooms into example region |
| 4. "Types that move together" | Covariance structure | Correlation heatmap or chord diagram |
| 5. "From covariance to prediction" | Bayesian update mechanism | Prior → poll → posterior diagram |
| 6. "How polls propagate" | Cross-state signal | Animated poll propagation on map |
| 7. "Validating the model" | Accuracy metrics | Holdout accuracy chart |
| 8. "What this means for 2026" | Closing connection to forecast | CTA to forecast hub |

**Step configuration from data file, not hardcoded JSX.** Each step: `{ id, text, vizType, vizConfig, transition }`. Model changes → update config + narrative text. No component surgery.

**`prefers-reduced-motion`:** Scroll triggers replaced with static sections. Content identical, not animated.

**`/methodology/accuracy`** — Traditional content page (not scrollytelling). Reference material. Rebuilt with shadcn + visx predicted-vs-actual scatter plot.

---

## Mobile Strategy

> **Research basis:** P6 — Mobile is a different product. The Economist and 538 completely restructure on mobile. See research §1 Principle 8. Anti-pattern: shrinking desktop layouts.

### Breakpoints

| Breakpoint | Width | Layout Strategy |
|------------|-------|-----------------|
| Desktop | ≥1024px | Full split-pane, side panels, hover tooltips |
| Tablet | 768–1023px | Stacked layout, panels below map, hover works |
| Mobile | <768px | Completely restructured experience |

### Mobile-Specific Transformations

| Component | Desktop | Mobile |
|-----------|---------|--------|
| Landing hero | Full layout | Vertical stack, mini-map hidden |
| Balance bar | Segmented interactive bar | "47D – 53R" text summary |
| Race cards | Grid | Horizontal swipeable carousel |
| "All Races" table | TanStack Table | Ranked list (no table) |
| Quantile dotplot | 100 dots | 50 dots |
| Poll tracker | Trend + confidence band | Trend line only + big number |
| Section weight sliders | Side-by-side | Full-width vertical stack |
| Scatter plot | Full interactive | Pinch-to-zoom, bottom sheet selectors |
| Type comparison | 4 columns | 2 columns |
| Map | Split pane | Full viewport, bottom sheet controls |
| Methodology | Sticky side viz | Inline viz between text blocks |
| Nav | Horizontal bar | Hamburger menu |
| Breadcrumbs | Full path | Truncated (first + last) |

### Touch Interaction Rules

These rules are codified and must be followed by all components:

1. **Tap = click** (primary action)
2. **Long-press = tooltip/detail** (replaces hover)
3. **Swipe = carousel navigation / sheet dismiss**
4. **Pinch = zoom** on maps and charts
5. **All interactive elements ≥ 44px** touch target
6. **No hover-only information** — everything accessible via tap or long-press
7. **No horizontal scroll tables** — use lists instead
8. **No multi-column comparison >2** on mobile
9. **No tiny polygon tap targets** at national zoom — use state level, then zoom for tracts
10. **No scroll hijacking** — react-scrollama uses IntersectionObserver, not scroll listeners

---

## Configuration Architecture

> **Research basis:** P5 — Configuration over hardcoding. P10 — No tech debt from day one. Audit finding: stale hardcoded "100 types / 5 super-types" contradicted the deployed J=130 / 6 super-type model.

Three config files decouple the frontend from model specifics. These are the **only** places non-API display metadata lives.

### 1. `web/lib/config/display.ts` — Field Display Metadata

Maps API field names to human-readable labels, format functions, sections, and sort order.

```typescript
export const FIELD_DISPLAY: Record<string, FieldConfig> = {
  median_hh_income: {
    label: "Median Household Income",
    format: "currency",
    section: "economics",
    sortOrder: 1,
  },
  pct_bachelors_plus: {
    label: "Bachelor's Degree+",
    format: "percent",
    section: "education",
    sortOrder: 2,
  },
  adherence_rate: {
    label: "Religious Adherence",
    format: "per1000_to_pct",  // Solves the RCMS gotcha permanently
    section: "culture",
    sortOrder: 10,
  },
  // New model features: add one entry. Removed features: delete one entry.
};
```

**Consumed by:** Type detail pages, county detail pages, comparison tables, tooltips, scatter plot axis labels.

**Rule:** If a field from the API is not in this config, render it with the raw API key as label and `toString()` as format. Never silently drop unknown fields.

### 2. `web/lib/config/palette.ts` — Color System

```typescript
export const RATING_COLORS = {
  safe_d: "#2d4a6f",
  likely_d: "#4b6d90",
  lean_d: "#7e9ab5",
  tossup: "#8a6b8a",   // Purple, NOT gray
  lean_r: "#c4907a",
  likely_r: "#9e5e4e",
  safe_r: "#6e3535",
} as const;

// Super-type colors — indexed by super_type_id from API
// When super-types grow from 6→8→10, add entries here
export const SUPER_TYPE_COLORS: [number, number, number][] = [
  [220, 120, 55],   // amber-orange
  [115, 45, 140],   // deep violet
  [220, 110, 110],  // rose-salmon
  [170, 35, 50],    // deep crimson
  [38, 145, 145],   // teal-cyan
  [195, 155, 25],   // deep gold
  [65, 140, 210],   // sky blue
  [40, 140, 85],    // emerald green
  // Add more as needed
];

export const CHOROPLETH = {
  demMin: 0.3,
  demMax: 0.7,
  steps: 9,
} as const;
```

**Rule:** Super-type names and count come from `/super-types` API. This file holds only colors and thresholds.

### 3. `web/lib/config/methodology.ts` — Scrollytelling Steps

```typescript
export const METHODOLOGY_STEPS: StepConfig[] = [
  {
    id: "intro",
    title: "What if elections aren't random?",
    text: "Beneath the noise of individual elections...",
    vizType: "map",
    vizConfig: { style: "stained-glass", zoom: "national" },
    transition: "fade",
  },
  // ... 7 more steps
];

export const MODEL_METRICS = {
  holdout_r: 0.698,
  loo_r: 0.711,
  rmse: 0.073,
  county_count: 3154,
  // Update these after each retrain
};
```

### The Zero-Hardcoding Rule

Any value that could change when the model retrains belongs in one of three places:

| Source | Examples | When It Changes |
|--------|----------|-----------------|
| API response | Type names, super-type count, demographic values, margins, poll data | Every retrain / poll ingest |
| Config file | Display labels, colors, thresholds, methodology text | Occasionally, by developer |
| Environment variable | API URL, feature flags | Per-deployment |

**Never in:** Component JSX, utility functions, scattered across files.

**Test:** After a model retrain that changes J from 130 to 150 and adds a new super-type, the frontend should require: zero code changes, at most one new color entry in `palette.ts`, and a redeploy.

---

## Data Layer

### SWR Hooks

Every API call goes through a custom SWR hook in `web/lib/hooks/`. Components never call `fetch` directly.

| Hook | Endpoint | Revalidation |
|------|----------|-------------|
| `useSenateOverview()` | `GET /senate/overview` | 5 min |
| `useGovernorOverview()` | `GET /governor/overview` | 5 min |
| `useRaceDetail(slug)` | `GET /races/{slug}` | 5 min |
| `useTypeDetail(id)` | `GET /types/{id}` | 30 min |
| `useCountyDetail(fips)` | `GET /counties/{fips}` | 30 min |
| `usePolls(params)` | `GET /polls` | 15 min |
| `useTypeScatterData()` | `GET /types/scatter-data` | 30 min |
| `useSuperTypes()` | `GET /super-types` | 60 min |
| `useForecast(params)` | `GET /forecast` | 5 min |

Each hook returns `{ data, error, isLoading }`. Error states render shadcn Alert with retry. Loading states render component-specific Skeleton.

### Error Boundaries

Every data-consuming section wraps in a React error boundary. Fallback: shadcn Alert — "Failed to load [section name]. Retrying..." with SWR automatic retry.

### Loading States — Skeleton Screens

> **Research basis:** P7 — Speed creates trust. See research §2 Pattern: Skeleton screens, pre-rendered static views.

Every component with async data has a Skeleton variant. The stained glass map gets a special treatment: gray polygon outlines fill with color progressively as tract data arrives state-by-state.

---

## Performance Architecture

| Strategy | Where | Why |
|----------|-------|-----|
| SSR at build time | Landing hero, race detail pages | First contentful paint <1s |
| ISR (revalidate: 300) | Forecast hub, race details | Fresh data without full rebuild |
| Client-side SWR | Polls, interactive map data | Real-time feel with caching |
| Dynamic import (ssr: false) | deck.gl MapShell, visx charts | Keep initial JS bundle small |
| Lazy load below fold | Comparison tables, methodology viz | Don't block above-fold content |
| Tract GeoJSON on demand | Per-state fetch on state click | 81K shapes cannot load at once |
| Simplified geometry | State boundaries at national zoom | 60fps pan/zoom |

---

## Accessibility (WCAG 2.1 AA)

> **Research basis:** See research §5 Anti-pattern 9 (low-contrast text), anti-pattern 13 (click targets too small).

**Rules:**

1. All text meets 4.5:1 contrast ratio against background (both themes)
2. All interactive elements have visible focus indicators (shadcn default)
3. All visx charts render a visually-hidden `<table>` with equivalent data
4. State-level keyboard navigation on map (arrow keys + Enter to drill in)
5. `prefers-reduced-motion`: all Motion animations disable, scrollytelling becomes static, map fly-to becomes instant
6. `prefers-color-scheme`: system theme detection (keep existing)
7. Skip-to-content link (keep existing)
8. ARIA landmarks on all major sections
9. Touch targets ≥ 44px on mobile

---

## Data Provenance Rules

> **Research basis:** P9. See research §1 Principle 14, §5 Anti-pattern 19.

Every data display must answer three questions:

1. **When?** — Timestamp: "Updated 2 hours ago" or absolute date
2. **How many?** — Source count: "Based on 7 polls" or "3,154 counties"
3. **How confident?** — Uncertainty: confidence interval, margin of error, or rating label

**No naked point estimates.** If a number appears without context, it's a bug.

---

## Code Quality Standards

These standards apply to every file in the new codebase:

1. Every component in its own file, max 200 lines. Past 200 → split.
2. All data-fetching via SWR hooks, never raw fetch in components
3. All display formatting via config system, never inline format logic
4. All colors via Tailwind tokens or palette config, never hex literals in JSX
5. TypeScript strict mode, no `any` types
6. Every interactive component has: loading Skeleton, error Alert, populated state
7. No `useEffect` for data fetching (SWR handles this)
8. No inline styles (entire point of the migration)
9. Components are pure renderers — data logic lives in hooks, display logic lives in config

---

## Testing Strategy

| Layer | Tool | What It Tests |
|-------|------|---------------|
| E2E critical flows | Playwright | Landing → forecast → race detail → back |
| Visual regression | Playwright screenshots | Stained glass map appearance |
| Component rendering | Vitest + RTL | Config-driven rendering: "new API field appears" |
| API contract | pytest (existing) | DuckDB → API chain integrity |
| Accessibility | axe-core in Playwright | WCAG 2.1 AA compliance per page |

---

## Anti-Patterns to Avoid

> **Research basis:** See research §5 — 27 anti-patterns documented.

| Anti-Pattern | Why It's Bad | Our Rule |
|--------------|-------------|----------|
| Gray for tossup | Reads as "no data" | Use purple (#8a6b8a) |
| SVG for 3K+ polygons | Sluggish rendering | deck.gl WebGL only |
| Point estimate without uncertainty | Misleading precision | Always show CI or distribution |
| No loading state | Users think it's broken | Skeleton on every async component |
| Hover-only information | Inaccessible on mobile | Tap/long-press alternatives |
| Partisan suffix in type names | Contradicts county-level data | Descriptive names only |
| Hardcoded model parameters | Stale after retrain | Config files or API |
| Horizontal scroll tables on mobile | Hostile UX | Lists on mobile |
| Continuous color gradient for uncertainty | Higher estimation error | Stepped gradient |
| Auto-playing animations | Disorienting | User-initiated or scroll-triggered only |

---

## Appendix: Research Sources

**Audit report:** `docs/frontend-audit-2026-03-28.md`
**Design research:** `research/political-dataviz-research.md`

**Key external references:**
- FiveThirtyEight graduated confidence shading + snake chart
- The Economist quantile dotplots ([GitHub model](https://github.com/TheEconomist/us-potus-model))
- NYT Upshot swing arrows + Bayesian needle
- Washington Post county shift dashboard + cartogram toggle
- 270toWin scenario builder + historical timeline
- Pudding.cool "MVP first" philosophy
- [Uncertainty visualization research (CHI 2018)](https://idl.cs.washington.edu/files/2018-UncertaintyBus-CHI.pdf)
- [Election forecast visualization research (Northwestern)](https://www.mccormick.northwestern.edu/news/articles/2024/07/building-informative-and-trustworthy-election-forecasts/)
- [Claus Wilke: Visualizing Uncertainty](https://clauswilke.com/dataviz/visualizing-uncertainty.html)
