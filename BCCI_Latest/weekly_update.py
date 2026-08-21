"""Incremental weekly refresh -- the entry point for the scheduled Action.

Three steps:

1. Refresh the match index for the current season (both classes).
2. Work out which matches need ball data, and fetch them.
3. Print a summary so the Action log says what actually changed.

Step 2 covers two cases beyond "never fetched":

*stale live scrapes*
    A match scraped while still in progress has only the balls bowled up to
    that moment. ``ball_coverage.csv`` records ``match_status`` at scrape time,
    so anything not stored as ``complete`` is re-fetched until it is.

*re-opened matches*
    A match whose index status moved on since we scraped it (abandoned ->
    complete, or a corrected scorecard) is re-fetched too.

Scope is international matches (men and women) from season 2026 onward. That is
a deliberate cut: domestic is ~90% of the volume at a far lower ball-data hit
rate, and older seasons are a one-off backfill rather than something a weekly
job should keep re-checking. Widen it with --level / --from-season.

Only ``Scorecard only`` coverage is treated as "nothing to fetch". ``Unknown``
is not -- probing found ~97% of Unknown internationals do return deliveries, so
filtering on the index's ``has_bbb`` flag would silently drop real data.

Usage:
    python weekly_update.py                      # maintained scope, 40/run
    python weekly_update.py --limit 0            # no cap
    python weekly_update.py --from-season 2020   # widen history
    python weekly_update.py --level international domestic
    python weekly_update.py --all-seasons        # also rebuild the whole index
"""

import argparse
import os

import pandas as pd

from fetch_balls import BALLS_DIR, COVERAGE_CSV, all_ball_files, scrape
from fetch_matches import DATA_DIR, update_index

MATCHES_CSV = os.path.join(DATA_DIR, "matches.csv")

# A scheduled run should finish in reasonable time even after a busy week.
DEFAULT_LIMIT = 40

# Maintained scope. Internationals only, current era onward -- domestic is ~90%
# of the volume for a much lower ball-data hit rate, and pre-2026 seasons are a
# one-off backfill rather than something a weekly job should keep re-checking.
DEFAULT_LEVELS = ("international",)
DEFAULT_FROM_SEASON = 2026

# 'Scorecard only' is the one coverage value that reliably means no ball data.
# 'Unknown' does not -- probing found ~97% of Unknown internationals do return
# deliveries -- so it must not be filtered out.
NO_BALL_DATA = "Scorecard only"


def load_coverage():
    """gid -> match_status recorded at scrape time."""
    if not os.path.exists(COVERAGE_CSV):
        return {}
    cov = pd.read_csv(COVERAGE_CSV, dtype=str, keep_default_na=False)
    if "match_status" not in cov.columns:
        return {gid: "" for gid in cov["gid"]}
    return dict(zip(cov["gid"], cov["match_status"]))


def on_disk():
    """gids already scraped, found anywhere under the format folders."""
    return {os.path.basename(f)[:-4] for f in all_ball_files()}


def in_scope(matches_df, levels=DEFAULT_LEVELS, from_season=DEFAULT_FROM_SEASON):
    """Completed matches inside the maintained scope that could have ball data."""
    df = matches_df[matches_df["source_status"] == "results"].copy()

    if levels:
        df = df[df["level"].isin(levels)]

    if from_season:
        years = pd.to_numeric(df.get("season_end_year"), errors="coerce")
        df = df[years >= from_season]

    # Drop only what definitely has nothing to fetch.
    return df[~df["coverage_detail"].str.startswith(NO_BALL_DATA, na=False)]


def pending(matches_df, limit=None, levels=DEFAULT_LEVELS, from_season=DEFAULT_FROM_SEASON):
    """Matches needing a ball fetch, newest first, split by reason."""
    df = in_scope(matches_df, levels, from_season)

    have = on_disk()
    coverage = load_coverage()

    missing, stale = [], []
    for _, row in df.sort_values("start_date", ascending=False).iterrows():
        gid = row["gid"]
        if gid not in have:
            missing.append(gid)
        elif coverage.get(gid, "") != "complete":
            # Scraped mid-match, or scraped before match_status was recorded.
            stale.append(gid)

    gids = missing + stale
    if limit:
        gids = gids[:limit]
    return gids, len(missing), len(stale)


def summarise():
    if not os.path.isdir(BALLS_DIR):
        return
    files = all_ball_files()
    if not files:
        return
    balls = sum(len(pd.read_csv(f, usecols=["scoring"], low_memory=False)) for f in files)
    print(f"\ncorpus: {len(files)} matches, {balls} deliveries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", help="seasons to refresh (default: current)")
    parser.add_argument("--all-seasons", action="store_true", help="full backfill")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"max matches to fetch this run (default {DEFAULT_LIMIT}; 0 = no cap)")
    parser.add_argument("--level", nargs="+", default=list(DEFAULT_LEVELS),
                        choices=("international", "domestic", "youth"),
                        help="which levels to maintain (default: international)")
    parser.add_argument("--from-season", type=int, default=DEFAULT_FROM_SEASON,
                        help=f"earliest season end year (default {DEFAULT_FROM_SEASON}; 0 = all)")
    parser.add_argument("--raw", action="store_true", help="keep raw JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1  match index")
    print("=" * 60)
    matches_df = update_index(seasons=args.season, all_seasons=args.all_seasons)

    if matches_df.empty:
        print("\nno matches in index -- nothing to do")
        return

    print("\n" + "=" * 60)
    print("STEP 2  ball-by-ball")
    print("=" * 60)
    scope = f"{'+'.join(args.level)}, season >= {args.from_season or 'all'}"
    gids, n_missing, n_stale = pending(
        matches_df, limit=args.limit or None,
        levels=args.level, from_season=args.from_season,
    )
    print(f"scope: {scope}")
    print(f"{n_missing} never fetched, {n_stale} incomplete when last fetched")

    if not gids:
        print("nothing to fetch -- corpus is up to date")
    else:
        if args.limit and n_missing + n_stale > args.limit:
            print(f"capped at {args.limit} this run; rerun to continue")
        # refresh=True so the stale ones actually get re-fetched rather than
        # skipped as already-present.
        scrape(gids, raw=args.raw, refresh=True)

    print("\n" + "=" * 60)
    print("STEP 3  summary")
    print("=" * 60)
    summarise()


if __name__ == "__main__":
    main()
