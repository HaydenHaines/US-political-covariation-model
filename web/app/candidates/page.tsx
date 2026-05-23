"use client";

/**
 * Candidates directory page (/candidates).
 *
 * Loads the full candidate list from the API via SWR (data is static between
 * pipeline runs), then applies client-side filtering as the user types.
 * This avoids unnecessary API round-trips for each keystroke.
 */

import { useEffect, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useCandidatesList } from "@/lib/hooks/use-candidates-list";
import type { CandidateListItem } from "@/lib/api";

// ── Filter + sort state ───────────────────────────────────────────────────────

type SortKey = "name" | "cec" | "n_races";

interface Filters {
  q: string;
  party: string;
  office: string;
  year: string;
  state: string;
  sort: SortKey;
}

// ── URL param names ───────────────────────────────────────────────────────────

const Q_PARAM = "q";
const PARTY_PARAM = "party";
const OFFICE_PARAM = "office";
const YEAR_PARAM = "year";
const STATE_PARAM = "state";
const SORT_PARAM = "sort";

// ── Default values ────────────────────────────────────────────────────────────

const DEFAULT_Q = "";
const DEFAULT_PARTY = "";
const DEFAULT_OFFICE = "";
const DEFAULT_YEAR = "";
const DEFAULT_STATE = "";
const DEFAULT_SORT: SortKey = "cec";

const INITIAL_FILTERS: Filters = {
  q: DEFAULT_Q,
  party: DEFAULT_PARTY,
  office: DEFAULT_OFFICE,
  year: DEFAULT_YEAR,
  state: DEFAULT_STATE,
  sort: DEFAULT_SORT,
};

// ── Validators ────────────────────────────────────────────────────────────────

function isParty(value: string | null): value is "" | "D" | "R" {
  return value === "" || value === "D" || value === "R";
}

function isOffice(value: string | null): value is "" | "Senate" | "Governor" {
  return value === "" || value === "Senate" || value === "Governor";
}

function isYear(value: string | null): value is string {
  if (value === null || value === "") return true;
  if (!/^\d{4}$/.test(value)) return false;
  const n = parseInt(value, 10);
  return n >= 2000 && n <= 2030;
}

function isState(value: string | null): value is string {
  return value === null || value === "" || /^[A-Z]{2}$/.test(value);
}

function isSortKey(value: string | null): value is SortKey {
  return value === "cec" || value === "n_races" || value === "name";
}

// ── URL helpers ───────────────────────────────────────────────────────────────

function setOrDelete(
  params: URLSearchParams,
  name: string,
  value: string,
  defaultValue: string,
): void {
  if (value !== defaultValue) {
    params.set(name, value);
  } else {
    params.delete(name);
  }
}

function getFiltersFromURL(sp: ReturnType<typeof useSearchParams>): Filters {
  const q = sp.get(Q_PARAM) ?? DEFAULT_Q;
  const rawParty = sp.get(PARTY_PARAM);
  const party = isParty(rawParty) ? rawParty : DEFAULT_PARTY;
  const rawOffice = sp.get(OFFICE_PARAM);
  const office = isOffice(rawOffice) ? rawOffice : DEFAULT_OFFICE;
  const rawYear = sp.get(YEAR_PARAM);
  const year = isYear(rawYear) ? (rawYear ?? DEFAULT_YEAR) : DEFAULT_YEAR;
  const rawState = sp.get(STATE_PARAM);
  const state = isState(rawState) ? (rawState ?? DEFAULT_STATE) : DEFAULT_STATE;
  const rawSort = sp.get(SORT_PARAM);
  const sort = isSortKey(rawSort) ? rawSort : DEFAULT_SORT;
  return { q, party, office, year, state, sort };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const PARTY_COLORS: Record<string, string> = {
  D: "var(--color-dem)",
  R: "var(--color-rep)",
};

function partyColor(party: string): string {
  return PARTY_COLORS[party] ?? "var(--color-text-muted)";
}

function formatCEC(cec: number): string {
  return `${Math.round(cec * 100)}%`;
}

// ── Candidate card ────────────────────────────────────────────────────────────

function CandidateCard({ candidate }: { candidate: CandidateListItem }) {
  const color = partyColor(candidate.party);

  return (
    <Link
      href={`/candidates/${candidate.bioguide_id}`}
      style={{ textDecoration: "none" }}
    >
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "6px",
          padding: "12px 14px",
          cursor: "pointer",
          transition: "border-color 0.15s",
        }}
        className="hover:border-[var(--color-text-muted)]"
      >
        {/* Name + party */}
        <div style={{ display: "flex", alignItems: "baseline", gap: "6px", marginBottom: "4px" }}>
          <span
            style={{
              fontFamily: "var(--font-serif)",
              fontWeight: 700,
              fontSize: "0.95rem",
              color,
            }}
          >
            {candidate.name}
          </span>
          <span style={{ fontSize: "0.7rem", color: "var(--color-text-muted)" }}>
            {candidate.party}
          </span>
        </div>

        {/* Office + states + years */}
        <div
          style={{
            fontSize: "0.72rem",
            color: "var(--color-text-muted)",
            marginBottom: "6px",
          }}
        >
          {candidate.offices.join(" / ")} · {candidate.states.join(", ")} ·{" "}
          {candidate.years.join(", ")}
        </div>

        {/* Stats row */}
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <span
            style={{
              fontSize: "0.65rem",
              fontWeight: 600,
              color: "var(--color-text-muted)",
              background: "rgba(148, 163, 184, 0.08)",
              border: "1px solid rgba(148, 163, 184, 0.20)",
              borderRadius: "4px",
              padding: "1px 5px",
            }}
          >
            {candidate.n_races} race{candidate.n_races !== 1 ? "s" : ""}
          </span>
          <span
            style={{
              fontSize: "0.65rem",
              fontWeight: 600,
              color: "var(--color-text-muted)",
              background: "rgba(148, 163, 184, 0.08)",
              border: "1px solid rgba(148, 163, 184, 0.20)",
              borderRadius: "4px",
              padding: "1px 5px",
            }}
          >
            CEC {formatCEC(candidate.cec)}
          </span>
        </div>

        {/* Badge pills (first 3) */}
        {candidate.badges.length > 0 && (
          <div
            style={{
              marginTop: "6px",
              display: "flex",
              flexWrap: "wrap",
              gap: "3px",
            }}
          >
            {candidate.badges.slice(0, 3).map((badge) => (
              <span
                key={badge}
                style={{
                  fontSize: "0.6rem",
                  padding: "1px 6px",
                  borderRadius: "9999px",
                  background: "rgba(148, 163, 184, 0.10)",
                  color: "var(--color-text-muted)",
                  border: "1px solid rgba(148, 163, 184, 0.25)",
                }}
              >
                {badge}
              </span>
            ))}
            {candidate.badges.length > 3 && (
              <span
                style={{
                  fontSize: "0.6rem",
                  color: "var(--color-text-subtle)",
                }}
              >
                +{candidate.badges.length - 3} more
              </span>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}

// ── Filter bar ────────────────────────────────────────────────────────────────

interface FilterBarProps {
  filters: Filters;
  onChange: (next: Partial<Filters>) => void;
  allYears: number[];
  allStates: string[];
}

function FilterBar({ filters, onChange, allYears, allStates }: FilterBarProps) {
  const inputStyle: React.CSSProperties = {
    background: "var(--color-surface)",
    border: "1px solid var(--color-border)",
    borderRadius: "4px",
    padding: "5px 8px",
    fontSize: "0.75rem",
    color: "var(--color-text)",
    outline: "none",
  };

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "8px",
        marginBottom: "16px",
      }}
    >
      {/* Name search */}
      <input
        type="text"
        placeholder="Search by name…"
        value={filters.q}
        onChange={(e) => onChange({ q: e.target.value })}
        style={{ ...inputStyle, minWidth: "180px", flex: "1 1 180px" }}
        aria-label="Search candidates by name"
      />

      {/* Party filter */}
      <select
        value={filters.party}
        onChange={(e) => onChange({ party: e.target.value })}
        style={{ ...inputStyle, minWidth: "90px" }}
        aria-label="Filter by party"
      >
        <option value="">All parties</option>
        <option value="D">Democrat</option>
        <option value="R">Republican</option>
      </select>

      {/* Office filter */}
      <select
        value={filters.office}
        onChange={(e) => onChange({ office: e.target.value })}
        style={{ ...inputStyle, minWidth: "110px" }}
        aria-label="Filter by office"
      >
        <option value="">All offices</option>
        <option value="Senate">Senate</option>
        <option value="Governor">Governor</option>
      </select>

      {/* Year filter */}
      <select
        value={filters.year}
        onChange={(e) => onChange({ year: e.target.value })}
        style={{ ...inputStyle, minWidth: "90px" }}
        aria-label="Filter by election year"
      >
        <option value="">All years</option>
        {allYears.map((y) => (
          <option key={y} value={String(y)}>
            {y}
          </option>
        ))}
      </select>

      {/* State filter */}
      <select
        value={filters.state}
        onChange={(e) => onChange({ state: e.target.value })}
        style={{ ...inputStyle, minWidth: "80px" }}
        aria-label="Filter by state"
      >
        <option value="">All states</option>
        {allStates.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>

      {/* Sort */}
      <select
        value={filters.sort}
        onChange={(e) => onChange({ sort: e.target.value as SortKey })}
        style={{ ...inputStyle, minWidth: "120px" }}
        aria-label="Sort candidates"
      >
        <option value="cec">Sort: CEC ↓</option>
        <option value="n_races">Sort: Most races</option>
        <option value="name">Sort: Name A–Z</option>
      </select>

      {/* Reset */}
      {(filters.q || filters.party || filters.office || filters.year || filters.state) && (
        <button
          onClick={() => onChange(INITIAL_FILTERS)}
          style={{
            ...inputStyle,
            cursor: "pointer",
            color: "var(--color-text-muted)",
          }}
          aria-label="Clear all filters"
        >
          Clear
        </button>
      )}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function CandidatesPage() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Load the full list once — client-side filtering avoids API round-trips per keystroke.
  const { data, isLoading, error } = useCandidatesList({});

  // Derive filters from URL params.
  const filters = useMemo(
    () => getFiltersFromURL(searchParams),
    [searchParams],
  );

  // Canonicalise stale/invalid/default URL params on mount and on searchParams change.
  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    let changed = false;

    const rawQ = searchParams.get(Q_PARAM);
    const rawParty = searchParams.get(PARTY_PARAM);
    const rawOffice = searchParams.get(OFFICE_PARAM);
    const rawYear = searchParams.get(YEAR_PARAM);
    const rawState = searchParams.get(STATE_PARAM);
    const rawSort = searchParams.get(SORT_PARAM);

    // q: delete if missing or equals default (empty string after trim)
    if (rawQ !== null) {
      if (rawQ.trim() === DEFAULT_Q) {
        params.delete(Q_PARAM);
        changed = true;
      }
    }

    // party: delete if invalid or default
    if (rawParty !== null) {
      if (!isParty(rawParty) || rawParty === DEFAULT_PARTY) {
        params.delete(PARTY_PARAM);
        changed = true;
      }
    }

    // office: delete if invalid or default
    if (rawOffice !== null) {
      if (!isOffice(rawOffice) || rawOffice === DEFAULT_OFFICE) {
        params.delete(OFFICE_PARAM);
        changed = true;
      }
    }

    // year: delete if invalid or default
    if (rawYear !== null) {
      if (!isYear(rawYear) || rawYear === DEFAULT_YEAR) {
        params.delete(YEAR_PARAM);
        changed = true;
      }
    }

    // state: delete if invalid or default
    if (rawState !== null) {
      if (!isState(rawState) || rawState === DEFAULT_STATE) {
        params.delete(STATE_PARAM);
        changed = true;
      }
    }

    // sort: delete if invalid or default
    if (rawSort !== null) {
      if (!isSortKey(rawSort) || rawSort === DEFAULT_SORT) {
        params.delete(SORT_PARAM);
        changed = true;
      }
    }

    if (!changed) return;
    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }, [pathname, router, searchParams, filters]);

  // Derive unique years and states from the full dataset for filter dropdowns.
  const { allYears, allStates } = useMemo(() => {
    if (!data) return { allYears: [], allStates: [] };
    const years = new Set<number>();
    const states = new Set<string>();
    for (const c of data.candidates) {
      c.years.forEach((y) => years.add(y));
      c.states.forEach((s) => states.add(s));
    }
    return {
      allYears: Array.from(years).sort((a, b) => b - a),
      allStates: Array.from(states).sort(),
    };
  }, [data]);

  // Apply client-side filters + sort on top of the full list.
  const filtered = useMemo(() => {
    if (!data) return [];
    const results = data.candidates.filter((c) => {
      if (filters.q && !c.name.toLowerCase().includes(filters.q.toLowerCase())) return false;
      if (filters.party && c.party !== filters.party) return false;
      if (filters.office && !c.offices.includes(filters.office)) return false;
      if (filters.year && !c.years.includes(Number(filters.year))) return false;
      if (filters.state && !c.states.includes(filters.state)) return false;
      return true;
    });
    if (filters.sort === "cec") results.sort((a, b) => b.cec - a.cec);
    else if (filters.sort === "n_races") results.sort((a, b) => b.n_races - a.n_races);
    else results.sort((a, b) => a.name.localeCompare(b.name));
    return results;
  }, [data, filters]);

  function handleFilterChange(next: Partial<Filters>) {
    const merged = { ...filters, ...next };
    const params = new URLSearchParams(searchParams.toString());
    setOrDelete(params, Q_PARAM, merged.q.trim(), DEFAULT_Q);
    setOrDelete(params, PARTY_PARAM, merged.party, DEFAULT_PARTY);
    setOrDelete(params, OFFICE_PARAM, merged.office, DEFAULT_OFFICE);
    setOrDelete(params, YEAR_PARAM, merged.year, DEFAULT_YEAR);
    setOrDelete(params, STATE_PARAM, merged.state, DEFAULT_STATE);
    setOrDelete(params, SORT_PARAM, merged.sort, DEFAULT_SORT);
    const query = params.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }

  return (
    <div style={{ maxWidth: "900px", margin: "0 auto", padding: "24px 16px" }}>
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "1.6rem",
          fontWeight: 700,
          marginBottom: "4px",
          color: "var(--color-text)",
        }}
      >
        Candidate Performance Profiles
      </h1>
      <p
        style={{
          fontSize: "0.82rem",
          color: "var(--color-text-muted)",
          marginBottom: "20px",
        }}
      >
        Sabermetric badges, election history, and CTOV fingerprints for Senate and Governor candidates.
      </p>

      {error && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "6px",
            background: "rgba(239, 68, 68, 0.08)",
            border: "1px solid rgba(239, 68, 68, 0.25)",
            color: "var(--color-text-muted)",
            fontSize: "0.82rem",
            marginBottom: "16px",
          }}
        >
          Failed to load candidates. Please try again.
        </div>
      )}

      <FilterBar
        filters={filters}
        onChange={handleFilterChange}
        allYears={allYears}
        allStates={allStates}
      />

      {isLoading ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
            gap: "10px",
          }}
        >
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={i}
              style={{
                height: "100px",
                borderRadius: "6px",
                background: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                animation: "pulse 1.5s ease-in-out infinite",
              }}
            />
          ))}
        </div>
      ) : (
        <>
          <p
            style={{
              fontSize: "0.72rem",
              color: "var(--color-text-subtle)",
              marginBottom: "12px",
            }}
          >
            {filtered.length} candidate{filtered.length !== 1 ? "s" : ""}
            {data && filtered.length < data.total ? ` (filtered from ${data.total})` : ""}
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
              gap: "10px",
            }}
          >
            {filtered.map((candidate) => (
              <CandidateCard key={candidate.bioguide_id} candidate={candidate} />
            ))}
          </div>
          {filtered.length === 0 && !isLoading && (
            <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem" }}>
              No candidates match the current filters.
            </p>
          )}
        </>
      )}
    </div>
  );
}
