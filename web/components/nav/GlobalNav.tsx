"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { MAIN_NAV } from "@/lib/config/navigation";
import { ThemeToggle } from "@/components/shared/ThemeToggle";
import { cn } from "@/lib/utils";

/**
 * Global navigation bar.
 *
 * Desktop (≥768px): horizontal link row.
 * Mobile (<768px): hamburger button that opens a slide-in panel from the right.
 */
export function GlobalNav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      <header className="sticky top-0 z-50 h-12 border-b border-[var(--color-border)] bg-[var(--color-bg)]/80 backdrop-blur-sm">
        <nav className="mx-auto flex h-full max-w-5xl items-center justify-between px-4">
          <Link
            href="/"
            className="font-serif text-lg font-bold tracking-tight no-underline"
            style={{ color: "var(--color-text)" }}
          >
            WetherVane
          </Link>

          {/* Desktop nav links (≥768px) */}
          <div className="hidden md:flex items-center gap-1">
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

          {/* Mobile: theme toggle + hamburger button (<768px) */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            <button
              onClick={() => setMobileOpen(true)}
              className="flex items-center justify-center rounded-md p-2 min-h-[44px] min-w-[44px]"
              style={{ color: "var(--color-text)" }}
              aria-label="Open navigation menu"
              aria-expanded={mobileOpen}
            >
              <Menu size={20} aria-hidden />
            </button>
          </div>
        </nav>
      </header>

      {/* Mobile: slide-in nav panel (<768px) */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-50 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-modal="true"
          role="dialog"
          aria-label="Navigation menu"
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/40" />

          {/* Panel — slides in from the right */}
          <nav
            className="absolute top-0 right-0 bottom-0 w-64 flex flex-col py-4"
            style={{
              background: "var(--color-bg)",
              borderLeft: "1px solid var(--color-border)",
              boxShadow: "-4px 0 24px rgba(0,0,0,0.15)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Panel header */}
            <div className="flex items-center justify-between px-4 mb-4">
              <span
                className="font-serif text-lg font-bold"
                style={{ color: "var(--color-text)" }}
              >
                WetherVane
              </span>
              <button
                onClick={() => setMobileOpen(false)}
                className="p-2 min-h-[44px] min-w-[44px] flex items-center justify-center rounded-md"
                style={{ color: "var(--color-text-muted)" }}
                aria-label="Close navigation menu"
              >
                <X size={20} aria-hidden />
              </button>
            </div>

            {/* Nav links */}
            {MAIN_NAV.map((item) => {
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "block px-4 py-3 text-base no-underline transition-colors min-h-[44px]",
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
          </nav>
        </div>
      )}
    </>
  );
}
