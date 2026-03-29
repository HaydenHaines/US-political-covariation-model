"use client";

import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";

const CYCLE: Theme[] = ["system", "light", "dark"];
const ICONS: Record<Theme, string> = { system: "◑", light: "☀", dark: "☾" };

/**
 * Theme toggle button: system → light → dark → system.
 * Stores preference in localStorage under 'wethervane-theme'.
 * Applies via data-theme attribute on <html> (read by CSS variables in globals.css).
 *
 * Note: Our CSS variables use hex values, so Tailwind's rgb() opacity syntax
 * won't work with them. Use opacity utilities for hover effects instead.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    const stored = localStorage.getItem("wethervane-theme") as Theme | null;
    if (stored && CYCLE.includes(stored)) setTheme(stored);
  }, []);

  function cycle() {
    const next = CYCLE[(CYCLE.indexOf(theme) + 1) % CYCLE.length];
    setTheme(next);
    localStorage.setItem("wethervane-theme", next);
    document.documentElement.setAttribute("data-theme", next);

    // Sync .dark class for shadcn components
    const shouldBeDark =
      next === "dark" ||
      (next === "system" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", shouldBeDark);
  }

  return (
    <button
      onClick={cycle}
      className="text-lg w-8 h-8 flex items-center justify-center rounded hover:opacity-70 transition-opacity"
      aria-label={`Theme: ${theme}. Click to cycle.`}
      title={`Theme: ${theme}`}
    >
      {ICONS[theme]}
    </button>
  );
}
