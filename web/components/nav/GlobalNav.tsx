"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MAIN_NAV } from "@/lib/config/navigation";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { cn } from "@/lib/utils";

export function GlobalNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 h-12 border-b border-[var(--color-border)] bg-[var(--color-bg)]/80 backdrop-blur-sm">
      <nav className="mx-auto flex h-full max-w-5xl items-center justify-between px-4">
        <Link
          href="/"
          className="font-serif text-lg font-bold tracking-tight no-underline"
          style={{ color: "var(--color-text)" }}
        >
          WetherVane
        </Link>

        <div className="flex items-center gap-1">
          {MAIN_NAV.map((item) => {
            const isActive =
              item.href === "/"
                ? pathname === "/"
                : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded px-3 py-1.5 text-sm no-underline transition-colors",
                  isActive
                    ? "font-semibold"
                    : "hover:bg-[var(--color-surface-raised)]",
                )}
                style={{
                  color: isActive
                    ? "var(--color-text)"
                    : "var(--color-text-muted)",
                }}
              >
                {item.label}
              </Link>
            );
          })}

          <div className="ml-2 border-l border-[var(--color-border)] pl-2">
            <ThemeToggle />
          </div>
        </div>
      </nav>
    </header>
  );
}
