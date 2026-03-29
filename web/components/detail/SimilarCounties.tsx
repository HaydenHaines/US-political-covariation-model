/**
 * SimilarCounties — list of same-type sibling counties.
 *
 * Each county links to /county/[fips]. Groups by state for
 * compact display when there are many siblings.
 */

import Link from "next/link";

interface SiblingCounty {
  county_fips: string;
  county_name: string | null;
  state_abbr: string;
}

interface SimilarCountiesProps {
  siblings: SiblingCounty[];
  typeName: string;
}

function stripStateSuffix(name: string | null): string {
  if (!name) return "Unknown County";
  return name.replace(/,\s*[A-Z]{2}$/, "");
}

export function SimilarCounties({ siblings, typeName }: SimilarCountiesProps) {
  if (siblings.length === 0) {
    return (
      <p style={{ color: "var(--color-text-muted)", fontSize: 14 }}>
        No similar counties found.
      </p>
    );
  }

  return (
    <div>
      <p
        style={{
          fontSize: 14,
          color: "var(--color-text-muted)",
          marginBottom: 12,
        }}
      >
        Other counties classified as <strong>{typeName}</strong>:
      </p>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "4px 16px",
        }}
      >
        {siblings.map((s) => (
          <Link
            key={s.county_fips}
            href={`/county/${s.county_fips}`}
            style={{
              display: "block",
              padding: "6px 0",
              fontSize: 14,
              color: "var(--color-dem)",
              textDecoration: "none",
              borderBottom: "1px solid var(--color-bg)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {stripStateSuffix(s.county_name)}, {s.state_abbr}
          </Link>
        ))}
      </div>
    </div>
  );
}
