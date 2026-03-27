"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { label: "Forecast", href: "/forecast" },
  { label: "Compare", href: "/compare" },
  { label: "Explore", href: "/explore" },
  { label: "About", href: "/about" },
];

export function TabBar() {
  const pathname = usePathname();
  return (
    <nav className="tab-bar" aria-label="Main navigation" style={{
      display: "flex",
      gap: "0",
      borderBottom: "1px solid var(--color-border)",
      background: "var(--color-surface)",
    }}>
      {TABS.map((tab) => {
        const active = pathname.startsWith(tab.href);
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className="tab-link"
            aria-current={active ? "page" : undefined}
            style={{
              padding: "10px 20px",
              fontFamily: "var(--font-serif)",
              fontSize: "14px",
              fontWeight: active ? "700" : "400",
              color: active ? "var(--color-text)" : "var(--color-text-muted)",
              borderBottom: active ? "2px solid var(--color-text)" : "2px solid transparent",
              textDecoration: "none",
              transition: "color 0.15s",
            }}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
