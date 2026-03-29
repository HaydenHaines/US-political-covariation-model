# Touch Interaction Rules

Governing rules for touch UX across WetherVane. Apply to all interactive UI elements on mobile (<768px).

---

## Rule 1 — Minimum touch target size: 44×44px

All tappable elements must have a minimum hit area of 44×44px, matching Apple HIG and WCAG 2.5.5.
Use `min-h-[44px] min-w-[44px]` Tailwind classes on buttons and links that would otherwise be too small.
Inline icon-only buttons are the most common violation — always pad them.

## Rule 2 — No hover-only interactions

Do not rely on CSS `:hover` or `onMouseEnter`/`onMouseLeave` for touch users. Any interaction that
reveals content on hover (tooltips, dropdown menus, descriptive labels) must have an equivalent tap
or long-press path.

## Rule 3 — Map tooltips: long-press to reveal

Map tooltips (MapTooltip) use hover on desktop. On touch devices, a long press (≥400ms) on a map
polygon or point should open the tooltip. Implement via `onTouchStart` + `setTimeout` with
`onTouchEnd` cancellation. Dismiss the tooltip with a tap outside.

## Rule 4 — Chart tooltips: long-press to reveal

ViSX/SVG chart tooltips that use `onMouseEnter`/`onMouseLeave` should also support long-press on
touch screens. Bind `onTouchStart` with a 400ms timeout on `<circle>` / `<rect>` elements. Cancel
on `onTouchEnd` (short tap) and dismiss on subsequent tap elsewhere.

## Rule 5 — Scroll-safe touch areas

Interactive elements inside scrollable containers (horizontal carousels, overflow-x tables) must not
intercept the scroll gesture. Prefer `onTouchEnd` over `onTouchStart` for click-equivalent actions
in scrollable contexts to allow the user to scroll without accidentally activating elements.

## Rule 6 — Bottom sheets over dropdown menus on mobile

When a control requires selecting from a list (axis selector, filter panel), prefer a bottom sheet
over a floating dropdown on mobile. Bottom sheets are easier to reach with a thumb, dismiss naturally
with a tap outside or swipe down, and do not clip near screen edges.

## Rule 7 — Snap-scroll carousels use CSS snap, not JS

Horizontal carousels use CSS `scroll-snap-type: x mandatory` and `scroll-snap-align: start` via
Tailwind classes `snap-x snap-mandatory` / `snap-start`. Do not implement custom touch tracking in
JavaScript — the browser's native scroll handling is smoother and respects momentum.

## Rule 8 — Hamburger menu closes on link tap

Mobile nav panels must close immediately when the user taps a navigation link. Pass
`onClick={() => setMobileOpen(false)}` to every link inside the panel. This prevents users from
having to manually close the panel after navigation.

## Rule 9 — Focus management for modal panels

When a bottom sheet or slide-in panel opens, programmatically move focus to the first interactive
element inside the panel (`useEffect` + `ref.current?.focus()`). When the panel closes, return focus
to the element that triggered it. This ensures keyboard and screen reader users are not lost.

## Rule 10 — Text summary fallbacks for dense visualizations

Visualizations that are too dense or too wide to be useful on small screens (stacked bar charts,
multi-column tables, 100-dot plots) must include a text-based fallback or a reduced-density variant
at `<768px`. Use `hidden md:block` / `md:hidden` Tailwind classes to show the appropriate version
without JavaScript. The text fallback must convey the same key insight as the visual.

---

## Breakpoints

| Prefix | Min width | Usage |
|--------|-----------|-------|
| (none) | 0px       | Mobile-first base styles |
| `md:`  | 768px     | Desktop enhancements |
| `lg:`  | 1024px    | Wide desktop |

## Implementation pattern

```tsx
{/* Mobile version */}
<div className="md:hidden">...</div>

{/* Desktop version */}
<div className="hidden md:block">...</div>
```

For touch targets:
```tsx
<button className="min-h-[44px] min-w-[44px] flex items-center justify-center p-2">
  <Icon aria-hidden />
</button>
```
