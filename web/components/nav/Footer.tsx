import Link from "next/link";
import { FOOTER_NAV } from "@/lib/config/navigation";

export function Footer() {
  return (
    <footer className="border-t border-[var(--color-border)] py-6">
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-4 px-4 sm:flex-row sm:justify-between">
        <nav className="flex flex-wrap gap-4">
          {FOOTER_NAV.map((item) =>
            item.external ? (
              <a
                key={item.href}
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm"
                style={{ color: "var(--color-text-muted)" }}
              >
                {item.label}
              </a>
            ) : (
              <Link
                key={item.href}
                href={item.href}
                className="text-sm"
                style={{ color: "var(--color-text-muted)" }}
              >
                {item.label}
              </Link>
            ),
          )}
        </nav>

        <p
          className="text-xs"
          style={{ color: "var(--color-text-subtle)" }}
        >
          WetherVane &mdash; Community-based electoral forecasting
        </p>
      </div>
    </footer>
  );
}
