"use client";

import { usePathname } from "next/navigation";
import { GlobalNav } from "@/components/nav/GlobalNav";
import { Footer } from "@/components/nav/Footer";

/**
 * Renders global site chrome (nav + footer) for all routes except embeds.
 *
 * Embed pages (/embed/*) are loaded inside third-party iframes and must not
 * include WetherVane navigation — only the widget card itself.
 *
 * This client component is the only place that needs to know about the
 * embed/non-embed distinction.  The root layout stays a Server Component.
 */
export function SiteChrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isEmbed = pathname.startsWith("/embed/") || pathname === "/embed";

  if (isEmbed) {
    return <>{children}</>;
  }

  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <GlobalNav />
      <main id="main-content" className="flex-1">
        {children}
      </main>
      <Footer />
    </>
  );
}
