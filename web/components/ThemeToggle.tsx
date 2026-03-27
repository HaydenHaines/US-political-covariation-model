"use client";

import { useState, useEffect, useCallback } from "react";

type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "wethervane-theme";

/**
 * Read the stored theme preference, falling back to "system".
 * Safe to call during SSR (returns "system").
 */
function getStoredTheme(): Theme {
  if (typeof window === "undefined") return "system";
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return "system";
}

/**
 * Apply the theme to the <html> element via data-theme attribute.
 * "system" lets the CSS `prefers-color-scheme` media query decide.
 */
function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
}

/**
 * Resolve the effective appearance ("light" or "dark") given the
 * current theme setting and system preference.
 */
function resolveAppearance(theme: Theme): "light" | "dark" {
  if (theme === "light" || theme === "dark") return theme;
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  const [appearance, setAppearance] = useState<"light" | "dark">("light");

  // Initialize from localStorage on mount
  useEffect(() => {
    const stored = getStoredTheme();
    setTheme(stored);
    applyTheme(stored);
    setAppearance(resolveAppearance(stored));
  }, []);

  // Listen for system preference changes when in "system" mode
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    function handleChange() {
      setAppearance(resolveAppearance(theme));
    }
    mq.addEventListener("change", handleChange);
    return () => mq.removeEventListener("change", handleChange);
  }, [theme]);

  const cycleTheme = useCallback(() => {
    // Cycle: system -> light -> dark -> system
    const order: Theme[] = ["system", "light", "dark"];
    const idx = order.indexOf(theme);
    const next = order[(idx + 1) % order.length];
    setTheme(next);
    applyTheme(next);
    setAppearance(resolveAppearance(next));
    localStorage.setItem(STORAGE_KEY, next);
  }, [theme]);

  // SVG icons for each state
  const icon = appearance === "dark" ? (
    // Moon icon (currently dark)
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  ) : (
    // Sun icon (currently light)
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );

  const label =
    theme === "system"
      ? "Theme: system preference"
      : theme === "dark"
        ? "Theme: dark mode"
        : "Theme: light mode";

  return (
    <button
      className="theme-toggle"
      onClick={cycleTheme}
      aria-label={label}
      title={label}
      type="button"
    >
      {icon}
    </button>
  );
}
