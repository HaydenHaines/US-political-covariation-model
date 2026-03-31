/**
 * Navigation and route configuration for WetherVane.
 *
 * Defines main nav items, footer links, and breadcrumb route mappings.
 * Components read from these configs instead of hardcoding routes.
 *
 * Update this file when:
 * - A new page or section is added
 * - Routes are restructured
 * - Nav labels change
 */

export interface NavItem {
  label: string;
  href: string;
  /** Whether this is an external link */
  external?: boolean;
}

/** Primary navigation items shown in the site header. */
export const MAIN_NAV: NavItem[] = [
  { label: "Forecast",    href: "/forecast" },
  { label: "Explore",     href: "/types" },
  { label: "Methodology", href: "/methodology" },
];

/** Footer navigation links. */
export const FOOTER_NAV: NavItem[] = [
  { label: "Forecast",    href: "/forecast" },
  { label: "Types",       href: "/types" },
  { label: "Shifts",      href: "/explore/shifts" },
  { label: "Methodology", href: "/methodology" },
  { label: "Compare",     href: "/compare" },
  { label: "Changelog",   href: "/changelog" },
  { label: "About",       href: "/methodology#about" },
  { label: "GitHub",      href: "https://github.com/HaydenHaines/wethervane", external: true },
];

/** Breadcrumb segment configuration. */
export interface BreadcrumbSegment {
  label: string;
  href: string;
}

/**
 * Route-to-breadcrumb mapping.
 *
 * Keys are route path prefixes. Values are arrays of breadcrumb segments
 * leading to (but not including) the current page. The current page is
 * appended dynamically by the breadcrumb component using page-specific data.
 */
export const BREADCRUMB_ROUTES: Record<string, BreadcrumbSegment[]> = {
  "/": [],
  "/forecast": [
    { label: "Home", href: "/" },
  ],
  "/forecast/[slug]": [
    { label: "Home", href: "/" },
    { label: "Forecast", href: "/forecast" },
  ],
  "/types": [
    { label: "Home", href: "/" },
  ],
  "/type/[id]": [
    { label: "Home", href: "/" },
    { label: "Types", href: "/types" },
  ],
  "/county/[fips]": [
    { label: "Home", href: "/" },
  ],
  "/changelog": [
    { label: "Home", href: "/" },
    { label: "Forecast", href: "/forecast" },
  ],
  "/methodology": [
    { label: "Home", href: "/" },
  ],
};

/**
 * Match a pathname to its breadcrumb route config.
 *
 * Handles both static routes ("/types") and dynamic routes ("/type/42")
 * by trying exact match first, then pattern match with bracket segments.
 */
export function getBreadcrumbs(pathname: string): BreadcrumbSegment[] {
  // Exact match
  if (BREADCRUMB_ROUTES[pathname]) {
    return BREADCRUMB_ROUTES[pathname];
  }

  // Dynamic route matching: /type/42 -> /type/[id]
  const segments = pathname.split("/").filter(Boolean);
  for (const pattern of Object.keys(BREADCRUMB_ROUTES)) {
    const patternSegments = pattern.split("/").filter(Boolean);
    if (patternSegments.length !== segments.length) continue;

    const matches = patternSegments.every(
      (ps, i) => ps.startsWith("[") || ps === segments[i],
    );
    if (matches) return BREADCRUMB_ROUTES[pattern];
  }

  // Fallback: just Home
  return [{ label: "Home", href: "/" }];
}
