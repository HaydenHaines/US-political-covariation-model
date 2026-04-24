# Poll Sources — Coverage Notes

Inventory of the poll scrapers that populate `data/polls/*.csv` on each
weekly refresh (`~/scripts/wethervane-poll-scrape.sh`).

## General-election polls — `polls_2026.csv`

Produced by `scripts/scrape_2026_polls.py`. Triple-sourced (in priority
order): **270toWin → RealClearPolling → Wikipedia**. The dedup layer keeps
the highest-priority source on any `(pollster, date, race)` key.

Schema: `race, geography, geo_level, dem_share, n_sample, date, pollster, notes`.
`dem_share` is the two-party Democratic fraction (`D / (D + R)`).

Covers 27 Senate + Governor 2026 races plus the generic congressional ballot.

## Primary polls — `primary_polls_2026.csv`

Produced by `scripts/scrape_rcp_primaries.py` (added 2026-04-21). **RCP only**
— primaries are within-party contests, so Wikipedia/270toWin tabular scraping
doesn't cleanly apply.

Schema: `race_key, geography, geo_level, party, date, pollster, n_sample,
candidates_json, is_primary, notes`. `candidates_json` is a JSON array of
`{"name": str, "pct": float}` sorted by pct descending (leader first).

RCP primary URL pattern: `/polls/{senate|governor}/{republican|democratic}-primary/{year}/{state}`.
**No candidate suffix** — unlike general-election matchup pages, primary pages
aggregate all polled candidates at the state-level URL.

Tracked 2026 primaries (verified against live RCP 2026-04-21):

| race_key                                  | URL                                                            |
|-------------------------------------------|----------------------------------------------------------------|
| 2026 GA Senate Republican Primary         | `/polls/senate/republican-primary/2026/georgia`               |
| 2026 TX Senate Republican Primary         | `/polls/senate/republican-primary/2026/texas`                 |
| 2026 TX Senate Democratic Primary         | `/polls/senate/democratic-primary/2026/texas`                 |
| 2026 IL Senate Democratic Primary         | `/polls/senate/democratic-primary/2026/illinois`              |
| 2026 GA Governor Republican Primary       | `/polls/governor/republican-primary/2026/georgia`             |
| 2026 OK Governor Republican Primary       | `/polls/governor/republican-primary/2026/oklahoma`            |
| 2026 SC Governor Republican Primary       | `/polls/governor/republican-primary/2026/south-carolina`      |
| 2026 RI Governor Democratic Primary       | `/polls/governor/democratic-primary/2026/rhode-island`        |

As RCP publishes more primaries, append entries to `PRIMARY_RACE_CONFIG` in
`scripts/scrape_rcp_primaries.py`. A 404/403 on a URL returns an empty list
with no side effects — safe to leave stale URLs in the config.

## Anti-bot notes

RCP serves a 403 to any request missing a realistic User-Agent. The fix
(S587, 2026-04-19) was a plain Chrome UA string + standard `Accept` and
`Accept-Language` headers; no JavaScript execution or proxy rotation needed.
Both scrapers inherit this via `fetch_html()` in `scrape_2026_polls.py`.

Request delay: 2 seconds between HTTP calls (`REQUEST_DELAY`). Do not
remove — we've had no further 403s under this cadence.

## Daily refresh wiring

`~/scripts/wethervane-poll-scrape.sh` runs both scrapers in sequence:

1. `scrape_2026_polls.py` — fatal on failure (general-election polls drive
   the production forecast).
2. `scrape_rcp_primaries.py` — non-fatal on failure (primaries are
   supplementary data; missing them should not block the weekly refresh).

Downstream: primary polls are not yet consumed by the prediction pipeline.
They exist as a data source for future matchup-accuracy improvements on
competitive primaries (e.g., weighting the D vs R general-election matchup
by the probability each primary candidate wins).
