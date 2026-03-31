import type { Metadata, Viewport } from "next";
import { SiteChrome } from "@/components/nav/SiteChrome";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL || "https://wethervane.hhaines.duckdns.org",
  ),
  title: "WetherVane — Electoral Forecast Model",
  description: "Community-based electoral forecasting for the 2026 midterms",
  alternates: {
    types: {
      "application/rss+xml": [
        { url: "/feed.xml", title: "WetherVane Forecast Updates" },
      ],
    },
  },
};

/**
 * Inline script that runs before React hydration to set the correct
 * theme attributes, preventing a flash of the wrong theme.
 *
 * Sets both data-theme (for WetherVane CSS variables) and .dark class
 * (for shadcn component dark mode).
 */
const THEME_INIT_SCRIPT = `
(function() {
  try {
    var stored = localStorage.getItem('wethervane-theme');
    if (stored === 'dark' || (stored !== 'light' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
      document.documentElement.setAttribute('data-theme', stored || 'system');
    } else {
      document.documentElement.setAttribute('data-theme', stored || 'system');
    }
  } catch (e) {
    document.documentElement.setAttribute('data-theme', 'system');
  }
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="flex flex-col">
        <SiteChrome>{children}</SiteChrome>
      </body>
    </html>
  );
}
