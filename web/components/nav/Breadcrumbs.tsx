"use client";

/**
 * Breadcrumbs navigation component.
 *
 * Reads parent segments from the navigation config based on the current
 * pathname, then appends the `currentPage` label as the terminal crumb.
 *
 * Mobile: shows only the last 2 crumbs (parent + current) to keep the
 * line from overflowing on small screens.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { getBreadcrumbs, type BreadcrumbSegment } from "@/lib/config/navigation";

interface BreadcrumbsProps {
  /** Label for the current (last) breadcrumb segment. */
  currentPage: string;
  /**
   * Optional extra crumb segments inserted between the route-derived parents
   * and the current page. Useful for adding geographic context (e.g. state)
   * that is only available from page-level data, not from the URL.
   */
  extraParents?: BreadcrumbSegment[];
}

export function Breadcrumbs({ currentPage, extraParents }: BreadcrumbsProps) {
  const pathname = usePathname();
  const routeParents: BreadcrumbSegment[] = getBreadcrumbs(pathname);
  const parents: BreadcrumbSegment[] = extraParents
    ? [...routeParents, ...extraParents]
    : routeParents;

  // Build the full crumb list: parents + current page
  const all: Array<{ label: string; href?: string }> = [
    ...parents.map((p) => ({ label: p.label, href: p.href })),
    { label: currentPage },
  ];

  // On mobile we show only the last 2 segments; all segments on larger screens.
  // We use a CSS class pair rather than JS slicing so the full list is in the DOM
  // (good for SEO / accessibility) — the truncation is purely visual.
  const showAllClass = "hidden sm:flex";
  const showLastTwoClass = "flex sm:hidden";

  const renderItems = (crumbs: typeof all) =>
    crumbs.map((crumb, idx) => {
      const isLast = idx === crumbs.length - 1;
      return (
        <li key={idx} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {idx > 0 && (
            <span aria-hidden="true" style={{ color: "var(--color-text-muted)" }}>
              /
            </span>
          )}
          {crumb.href && !isLast ? (
            <Link
              href={crumb.href}
              style={{ color: "var(--color-dem)", textDecoration: "none" }}
            >
              {crumb.label}
            </Link>
          ) : (
            <span aria-current={isLast ? "page" : undefined}>
              {crumb.label}
            </span>
          )}
        </li>
      );
    });

  // Visible on sm+ — all crumbs
  const allCrumbs = all;
  // Visible on mobile — last 2 crumbs only
  const mobileCrumbs = all.slice(-2);

  return (
    <nav
      aria-label="Breadcrumb"
      style={{ fontSize: 13, color: "var(--color-text-muted)", marginBottom: 24 }}
    >
      {/* Full breadcrumbs (hidden on mobile) */}
      <ol
        className={showAllClass}
        style={{
          listStyle: "none",
          margin: 0,
          padding: 0,
          flexWrap: "wrap",
          alignItems: "center",
          gap: "0 4px",
        }}
      >
        {renderItems(allCrumbs)}
      </ol>

      {/* Truncated breadcrumbs — last 2 only (visible on mobile) */}
      <ol
        className={showLastTwoClass}
        style={{
          listStyle: "none",
          margin: 0,
          padding: 0,
          alignItems: "center",
          gap: "0 4px",
        }}
      >
        {mobileCrumbs.length < allCrumbs.length && (
          <li style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span aria-hidden="true">…</span>
            <span aria-hidden="true" style={{ color: "var(--color-text-muted)" }}>
              /
            </span>
          </li>
        )}
        {renderItems(mobileCrumbs)}
      </ol>
    </nav>
  );
}
