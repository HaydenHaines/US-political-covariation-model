"use client";

import { useState, useEffect } from "react";

/**
 * Returns true when the given CSS media query matches.
 * Defaults to `false` on the server (SSR safe).
 *
 * @example
 *   const isMobile = useMediaQuery("(max-width: 767px)");
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(query);
    setMatches(mql.matches);

    function handleChange(e: MediaQueryListEvent) {
      setMatches(e.matches);
    }

    mql.addEventListener("change", handleChange);
    return () => mql.removeEventListener("change", handleChange);
  }, [query]);

  return matches;
}
