#!/usr/bin/env python3
"""
polymarket_divergence.py — Compare WetherVane predictions against Polymarket
implied probabilities for 2026 Senate and Governor races.

Run this manually when model output smells off, after a retrain, or when you
want to sense-check WetherVane against prediction market consensus.

Usage:
    uv run --with requests python3 scripts/polymarket_divergence.py
    uv run --with requests python3 scripts/polymarket_divergence.py --senate-only
    uv run --with requests python3 scripts/polymarket_divergence.py --governor-only
    uv run --with requests python3 scripts/polymarket_divergence.py --min-volume 10000

What this compares:
    WetherVane produces a vote-share margin (e.g. D+3.2pp = 0.532 dem two-party
    vote share). Polymarket produces a win *probability* (e.g. 62% chance Dems win).

    Vote share ≠ win probability. A model can say D+3pp (53%) with only 60% win
    probability because of uncertainty; another can say D+3pp with 70% win prob
    because of tighter uncertainty bands. We bridge the gap by converting WetherVane
    margins to an *implied* win probability via a calibrated logistic:
        wv_win_prob = 1 / (1 + exp(-LOGISTIC_K * margin))
    with LOGISTIC_K ≈ 11, which maps:
        D+10pp → ~75% win prob
        D+20pp → ~90% win prob
        tossup  → 50%
        R+10pp → ~25% win prob

    Divergence is then |wv_win_prob - pm_dem_win_prob|, expressed in probability
    points. A value >0.20 (20pp) is worth looking at.

    Direction disagreement (WV says D, Polymarket says R) is a much stronger
    signal — flagged with *** in the output.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

# ── Constants ─────────────────────────────────────────────────────────────────

WETHERVANE_API = "https://wethervane.hhaines.duckdns.org/api/v1"
POLYMARKET_EVENTS_API = "https://gamma-api.polymarket.com/events"
TIMEOUT = 20

# Logistic scale factor: tuned so D+10pp → ~75% win prob.
# Increase to make the curve steeper (less uncertainty assumed).
LOGISTIC_K = 11.0

# Minimum Polymarket market volume (USD) to trust a race.
# Low-volume markets can have stale/arbitrary prices.
DEFAULT_MIN_VOLUME = 5_000

# ANSI colors for terminal output
RED = "\033[91m"
YEL = "\033[93m"
GRN = "\033[92m"
CYN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RST = "\033[0m"

# State name ↔ abbreviation mapping
_STATE_NAME_TO_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WVRace:
    race_id: str        # e.g. "2026 WI Senate"
    state: str          # e.g. "WI"
    race_type: str      # "senate" or "governor"
    rating: str         # safe_d / likely_d / lean_d / tossup / lean_r / likely_r / safe_r
    margin: float       # dem_share - 0.5, positive = D favored
    n_polls: int

    @property
    def margin_pp(self) -> float:
        return self.margin * 100

    @property
    def implied_win_prob(self) -> float:
        """WetherVane margin converted to implied Dem win probability via logistic."""
        return 1.0 / (1.0 + math.exp(-LOGISTIC_K * self.margin))

    @property
    def direction(self) -> str:
        if self.margin > 0.005:
            return "D"
        elif self.margin < -0.005:
            return "R"
        return "TOSS"


@dataclass
class PMRace:
    event_id: str
    event_title: str
    state: str
    race_type: str      # "senate" or "governor"
    dem_win_prob: float # 0–1
    volume: float       # USD traded

    @property
    def direction(self) -> str:
        if self.dem_win_prob > 0.525:
            return "D"
        elif self.dem_win_prob < 0.475:
            return "R"
        return "TOSS"


@dataclass
class ComparedRace:
    wv: WVRace
    pm: PMRace
    divergence: float   # |wv_win_prob - pm_dem_win_prob|, in probability points

    @property
    def direction_agrees(self) -> bool:
        wv_d = self.wv.direction
        pm_d = self.pm.direction
        # Both tossup counts as agreement
        if wv_d == "TOSS" or pm_d == "TOSS":
            return True
        return wv_d == pm_d


# ── Polymarket fetching ───────────────────────────────────────────────────────

def _fetch_polymarket_events(race_type: str) -> list[dict]:
    """Fetch all open Polymarket events matching 'Senate Election Winner' or
    'Governor Election Winner'. Paginates until exhausted."""
    keyword = "Senate Election Winner" if race_type == "senate" else "Governor Election Winner"
    events: list[dict] = []
    offset = 0
    limit = 100
    while True:
        r = requests.get(
            POLYMARKET_EVENTS_API,
            params={"tag": "politics", "closed": "false", "limit": limit, "offset": offset},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for e in batch:
            if keyword in e.get("title", ""):
                events.append(e)
        if len(batch) < limit:
            break
        offset += limit
    return events


def _extract_dem_win_prob(markets: list[dict]) -> tuple[float, float] | None:
    """Return (dem_win_prob, volume) from the Dem-win market in an event's
    markets list. Returns None if no price data available."""
    for m in markets:
        q = m.get("question", "")
        if "Democrats" not in q and "Democratic" not in q:
            continue
        prices_raw = m.get("outcomePrices", "[]")
        try:
            import json as _json
            prices = [float(p) for p in _json.loads(prices_raw)]
        except Exception:
            continue
        if not prices:
            continue
        # prices[0] = P(Yes = Dems win), prices[1] = P(No)
        return prices[0], float(m.get("volume", 0) or 0)
    return None


def _parse_state_from_title(title: str, race_type: str) -> str | None:
    """Extract state abbreviation from a title like 'Colorado Senate Election Winner'."""
    suffix = " Senate Election Winner" if race_type == "senate" else " Governor Election Winner"
    state_name = title.replace(suffix, "").strip()
    return _STATE_NAME_TO_ABBR.get(state_name)


def fetch_polymarket_races(race_type: str, min_volume: float) -> dict[str, PMRace]:
    """Return {state_abbr: PMRace} for all Polymarket 2026 races of a given type."""
    events = _fetch_polymarket_events(race_type)
    races: dict[str, PMRace] = {}
    for event in events:
        state = _parse_state_from_title(event.get("title", ""), race_type)
        if not state:
            continue
        result = _extract_dem_win_prob(event.get("markets", []))
        if result is None:
            continue
        dem_prob, volume = result
        if volume < min_volume:
            continue
        races[state] = PMRace(
            event_id=str(event["id"]),
            event_title=event.get("title", ""),
            state=state,
            race_type=race_type,
            dem_win_prob=dem_prob,
            volume=volume,
        )
    return races


# ── WetherVane fetching ───────────────────────────────────────────────────────

def fetch_wethervane_races(race_type: str) -> dict[str, WVRace]:
    """Return {state_abbr: WVRace} for all WetherVane predictions of a given type."""
    endpoint = f"{WETHERVANE_API}/{race_type}/overview"
    r = requests.get(endpoint, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    races: dict[str, WVRace] = {}
    for item in data.get("races", []):
        state = item["state"]
        races[state] = WVRace(
            race_id=item["race"],
            state=state,
            race_type=race_type,
            rating=item["rating"],
            margin=item["margin"],
            n_polls=item.get("n_polls", 0),
        )
    return races


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_races(
    wv_races: dict[str, WVRace],
    pm_races: dict[str, PMRace],
) -> tuple[list[ComparedRace], list[WVRace], list[PMRace]]:
    """Join WV and PM by state. Returns (matched, wv_only, pm_only)."""
    matched: list[ComparedRace] = []
    wv_only: list[WVRace] = []
    pm_only: list[PMRace] = []

    for state, wv in wv_races.items():
        if state in pm_races:
            pm = pm_races[state]
            divergence = abs(wv.implied_win_prob - pm.dem_win_prob)
            matched.append(ComparedRace(wv=wv, pm=pm, divergence=divergence))
        else:
            wv_only.append(wv)

    for state, pm in pm_races.items():
        if state not in wv_races:
            pm_only.append(pm)

    matched.sort(key=lambda c: c.divergence, reverse=True)
    return matched, wv_only, pm_only


# ── Output formatting ─────────────────────────────────────────────────────────

def _rating_short(rating: str) -> str:
    return {
        "safe_d": "Safe D",
        "likely_d": "Likely D",
        "lean_d": "Lean D",
        "tossup": "TOSS",
        "lean_r": "Lean R",
        "likely_r": "Likely R",
        "safe_r": "Safe R",
    }.get(rating, rating)


def _divergence_color(divergence: float, agrees: bool) -> str:
    if not agrees:
        return RED
    if divergence >= 0.20:
        return YEL
    if divergence >= 0.10:
        return CYN
    return GRN


def _format_margin(margin_pp: float) -> str:
    sign = "D+" if margin_pp >= 0 else "R+"
    return f"{sign}{abs(margin_pp):.1f}pp"


def _format_prob(p: float) -> str:
    return f"{p*100:.0f}%"


def print_report(
    race_type: str,
    matched: list[ComparedRace],
    wv_only: list[WVRace],
    pm_only: list[PMRace],
    min_volume: float,
) -> None:
    label = race_type.capitalize()
    print()
    print(f"{BOLD}{'═'*72}{RST}")
    print(f"{BOLD}  WetherVane vs Polymarket — 2026 {label} Races{RST}")
    print(f"{DIM}  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
          f"Min PM volume: ${min_volume:,.0f}{RST}")
    print(f"{'═'*72}")

    if not matched:
        print(f"  {YEL}No matched races found.{RST}")
        return

    n_agree = sum(1 for c in matched if c.direction_agrees)
    mean_div = sum(c.divergence for c in matched) / len(matched)
    n_disagree = len(matched) - n_agree

    # ── Summary line ──────────────────────────────────────────────────────────
    agree_pct = n_agree / len(matched) * 100
    agree_color = GRN if agree_pct >= 80 else YEL if agree_pct >= 60 else RED
    print(f"\n  Matched {len(matched)} races  |  "
          f"Direction agreement: {agree_color}{n_agree}/{len(matched)} "
          f"({agree_pct:.0f}%){RST}  |  "
          f"Mean divergence: {mean_div*100:.1f}pp")
    if n_disagree:
        print(f"  {RED}{BOLD}Direction disagreements: {n_disagree} race(s) — see *** rows{RST}")

    # ── Table header ──────────────────────────────────────────────────────────
    print()
    col_race   = 26
    col_rating = 10
    col_margin = 10
    col_wv_p   = 9
    col_pm_p   = 9
    col_dir    = 6
    col_div    = 9
    col_polls  = 7
    col_vol    = 11

    hdr = (
        f"{'Race':<{col_race}}"
        f"{'WV Rating':<{col_rating}}"
        f"{'WV Margin':<{col_margin}}"
        f"{'WV Win%':<{col_wv_p}}"
        f"{'PM Win%':<{col_pm_p}}"
        f"{'Dir':<{col_dir}}"
        f"{'Diverg':<{col_div}}"
        f"{'Polls':<{col_polls}}"
        f"{'PM Vol':>{col_vol}}"
    )
    print(f"  {BOLD}{hdr}{RST}")
    print(f"  {'─'*72}")

    for c in matched:
        direction_flag = "   " if c.direction_agrees else f"{RED}***{RST}"
        color = _divergence_color(c.divergence, c.direction_agrees)
        polls_str = str(c.wv.n_polls) if c.wv.n_polls > 0 else f"{DIM}  0{RST}"
        vol_str = f"${c.pm.volume:>8,.0f}"

        row = (
            f"{c.wv.race_id:<{col_race}}"
            f"{_rating_short(c.wv.rating):<{col_rating}}"
            f"{_format_margin(c.wv.margin_pp):<{col_margin}}"
            f"{_format_prob(c.wv.implied_win_prob):<{col_wv_p}}"
            f"{color}{_format_prob(c.pm.dem_win_prob):<{col_pm_p}}{RST}"
            f"{direction_flag}"
            f"{color}{c.divergence*100:>6.1f}pp{RST}  "
            f"{polls_str:<{col_polls}}"
            f"{DIM}{vol_str}{RST}"
        )
        print(f"  {row}")

    # ── Unmatched ─────────────────────────────────────────────────────────────
    if wv_only:
        states = ", ".join(r.state for r in sorted(wv_only, key=lambda r: r.state))
        print(f"\n  {DIM}WetherVane only (no PM market or vol < ${min_volume:,.0f}): {states}{RST}")
    if pm_only:
        states = ", ".join(r.state for r in sorted(pm_only, key=lambda r: r.state))
        print(f"  {DIM}Polymarket only (not in WetherVane): {states}{RST}")

    # ── Biggest disagreements callout ─────────────────────────────────────────
    big = [c for c in matched if not c.direction_agrees or c.divergence >= 0.20]
    if big:
        print()
        print(f"  {BOLD}── Races worth investigating ─────────────────────────────────────{RST}")
        for c in big:
            flag = f"{RED}DIRECTION FLIP{RST}" if not c.direction_agrees else f"{YEL}large divergence{RST}"
            wv_str = f"WV: {_rating_short(c.wv.rating)} ({_format_margin(c.wv.margin_pp)}, "
            wv_str += f"implied {_format_prob(c.wv.implied_win_prob)} win)"
            pm_str = f"PM: {_format_prob(c.pm.dem_win_prob)} Dem win"
            polls = f"  [{c.wv.n_polls} polls]" if c.wv.n_polls else "  [no polls]"
            print(f"\n  {BOLD}{c.wv.race_id}{RST}  {flag}")
            print(f"    {wv_str}")
            print(f"    {pm_str}{polls}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare WetherVane vs Polymarket for 2026 races."
    )
    parser.add_argument("--senate-only", action="store_true")
    parser.add_argument("--governor-only", action="store_true")
    parser.add_argument(
        "--min-volume", type=float, default=DEFAULT_MIN_VOLUME,
        help=f"Minimum Polymarket USD volume to include a market (default: {DEFAULT_MIN_VOLUME:,})"
    )
    args = parser.parse_args()

    race_types = []
    if args.senate_only:
        race_types = ["senate"]
    elif args.governor_only:
        race_types = ["governor"]
    else:
        race_types = ["senate", "governor"]

    had_error = False
    for race_type in race_types:
        try:
            print(f"  Fetching WetherVane {race_type} predictions...", end=" ", flush=True)
            wv = fetch_wethervane_races(race_type)
            print(f"{len(wv)} races")

            print(f"  Fetching Polymarket {race_type} markets...", end=" ", flush=True)
            pm = fetch_polymarket_races(race_type, args.min_volume)
            print(f"{len(pm)} markets (min vol ${args.min_volume:,.0f})")

            matched, wv_only, pm_only = compare_races(wv, pm)
            print_report(race_type, matched, wv_only, pm_only, args.min_volume)

        except requests.RequestException as e:
            print(f"\n  {RED}ERROR fetching {race_type} data: {e}{RST}", file=sys.stderr)
            had_error = True

    print()
    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
