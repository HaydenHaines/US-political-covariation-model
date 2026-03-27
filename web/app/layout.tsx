import type { Metadata, Viewport } from "next";
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
  title: "WetherVane — 2026 Electoral Forecast",
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
 * data-theme attribute, preventing a flash of the wrong theme.
 */
const THEME_INIT_SCRIPT = `
(function() {
  try {
    var stored = localStorage.getItem('wethervane-theme');
    if (stored === 'light' || stored === 'dark') {
      document.documentElement.setAttribute('data-theme', stored);
    } else {
      document.documentElement.setAttribute('data-theme', 'system');
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
      <body>
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}
