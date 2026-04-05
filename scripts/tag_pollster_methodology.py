"""
Tag each poll row in polls_2026.csv with the pollster's primary survey methodology.

Methodology values:
  phone   - live telephone interviewer
  online  - online panel / web survey
  IVR     - interactive voice response (robo-poll)
  mixed   - combination of methodologies (e.g. IVR + online)
  unknown - methodology not confirmed

Run:
    python scripts/tag_pollster_methodology.py [--csv data/polls/polls_2026.csv]
"""

import csv
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Methodology map: each entry is (substring_pattern, methodology).
# Matching is case-insensitive and uses substring search.
# Entries are evaluated in order; the FIRST match wins.
# ---------------------------------------------------------------------------
METHODOLOGY_MAP = [
    # --- Phone (live caller) ---
    ("quinnipiac",          "phone"),
    ("marist",              "phone"),
    ("siena",               "phone"),
    ("fox news",            "phone"),
    ("beacon research",     "phone"),
    ("monmouth",            "phone"),
    ("cnn",                 "phone"),
    ("mason-dixon",         "phone"),
    ("mitchell research",   "phone"),
    ("epic-mra",            "phone"),
    ("marquette law",       "phone"),
    ("suffolk",             "phone"),   # Suffolk/USA Today, Boston Globe/Suffolk
    ("rmg research",        "phone"),
    ("fabrizio",            "phone"),   # Fabrizio Lee, Fabrizio McLaughlin
    ("franklin & marshall", "phone"),
    ("franklin &amp; marshall", "phone"),
    ("glengariff",          "phone"),
    ("pan atlantic",        "phone"),   # Pan Atlantic Research
    ("hart research",       "phone"),
    ("gbao",                "phone"),
    ("normington",          "phone"),
    ("schoen cooperman",    "phone"),
    ("saint anselm",        "phone"),
    ("st. anselm",          "phone"),
    ("unh",                 "phone"),   # UNH / University of New Hampshire
    ("university of new hampshire", "phone"),
    ("univ. of new hampshire",      "phone"),
    ("nh journal",          "phone"),
    ("nhjournal",           "phone"),
    ("university of north florida", "phone"),
    ("university of houston", "phone"),
    ("univ. of houston",    "phone"),
    ("ut tyler",            "phone"),
    ("univ. of texas",      "phone"),
    ("university of texas", "phone"),
    ("texas public opinion", "phone"),
    ("catawba college",     "phone"),
    ("bowling green",       "online"),  # Bowling Green State Univ / YouGov partnership
    ("detroit news",        "phone"),
    ("harvard-harris",      "phone"),
    ("npr",                 "phone"),
    ("reuters",             "phone"),
    ("platform communications", "phone"),
    ("emc research",        "phone"),
    ("emr",                 "phone"),

    # --- IVR ---
    ("public policy polling", "IVR"),
    ("insideradvantage",    "IVR"),
    ("insider advantage",   "IVR"),
    ("cygnal",              "IVR"),     # Cygnal uses IVR + text

    # --- Online ---
    ("morning consult",     "online"),
    ("yougov",              "online"),  # YouGov, Economist/YouGov, Bowling Green/YouGov, etc.
    ("echelon insights",    "online"),
    ("redfield",            "online"),
    ("data for progress",   "online"),
    ("civiqs",              "online"),
    ("atlasinte",           "online"),
    ("co/efficient",        "online"),
    ("change research",     "online"),
    ("decision desk",       "online"),
    ("aif center",          "online"),
    ("big data poll",       "online"),
    ("target insyght",      "online"),
    ("targoz",              "online"),
    ("victory insights",    "online"),
    ("quantus insights",    "online"),
    ("workbench strategy",  "online"),
    ("wpa intelligence",    "online"),
    ("praecones analytica", "online"),

    # --- Mixed ---
    ("emerson college",     "mixed"),   # IVR + online panel
    ("surveyusa",           "mixed"),   # IVR + online
    ("trafalgar",           "mixed"),   # IVR + online
    ("rasmussen",           "mixed"),   # IVR + online
    ("susquehanna",         "mixed"),   # phone + online
    ("noble predictive",    "mixed"),   # IVR + online
    ("tipp insights",       "mixed"),   # online + phone
    ("tyson group",         "mixed"),
    ("onmessage",           "mixed"),   # automated + online
    ("rosetta stone",       "mixed"),
    ("bendixen",            "mixed"),
    ("nexus strategies",    "mixed"),
    ("nexus/strategic",     "mixed"),
    ("plymouth union",      "mixed"),
    ("data targeting",      "mixed"),
    ("zenith research",     "online"),
    ("impact research",     "online"),
]


def lookup_methodology(pollster: str) -> str:
    """Return the methodology for the given pollster string."""
    lower = pollster.lower()
    for pattern, method in METHODOLOGY_MAP:
        if pattern.lower() in lower:
            return method
    return "unknown"


def tag_csv(csv_path: str) -> dict:
    """
    Read the CSV at csv_path, add/update the 'methodology' column, write back.
    Returns a summary dict.
    """
    path = Path(csv_path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Ensure methodology column exists
    if "methodology" not in fieldnames:
        fieldnames = list(fieldnames) + ["methodology"]

    tagged_counts: dict[str, int] = {}
    unknown_pollsters: set[str] = set()

    for row in rows:
        method = lookup_methodology(row["pollster"])
        row["methodology"] = method
        tagged_counts[method] = tagged_counts.get(method, 0) + 1
        if method == "unknown":
            unknown_pollsters.add(row["pollster"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total_rows": len(rows),
        "methodology_counts": tagged_counts,
        "unknown_pollsters": sorted(unknown_pollsters),
    }


def main():
    parser = argparse.ArgumentParser(description="Tag polls CSV with pollster methodology")
    parser.add_argument(
        "--csv",
        default="data/polls/polls_2026.csv",
        help="Path to polls CSV file (default: data/polls/polls_2026.csv)",
    )
    args = parser.parse_args()

    summary = tag_csv(args.csv)

    print(f"Tagged {summary['total_rows']} rows in {args.csv}")
    print("\nMethodology breakdown:")
    for method, count in sorted(summary["methodology_counts"].items()):
        print(f"  {method:10s}: {count}")

    if summary["unknown_pollsters"]:
        print(f"\nUnknown pollsters ({len(summary['unknown_pollsters'])}):")
        for p in summary["unknown_pollsters"]:
            print(f"  {p!r}")
    else:
        print("\nAll pollsters matched — no unknowns.")


if __name__ == "__main__":
    main()
