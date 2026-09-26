"""Incremental refresh, run by hand during the IPL season.

1. Refresh the IPL match index for the current season.
2. Fetch ball data for matches never fetched, or fetched before they finished.
3. Print a summary of what actually changed.

A match scraped mid-innings holds only the balls bowled so far, so
``ball_coverage.csv`` records ``match_status`` at scrape time and anything short
of ``complete`` is re-fetched until it settles.

Only the current season is re-checked by default. Older seasons are a one-off
backfill (``--all-seasons --from-season 0 --limit 0``), not something a weekly
job needs to walk -- and abandoned matches with no commentary would otherwise be
retried every week forever.

Usage:
    python weekly_update.py                       # current season, 40/run
    python weekly_update.py --limit 0             # no cap
    python weekly_update.py --all-seasons --from-season 0 --limit 0   # backfill
"""

import argparse
import os
import time

import pandas as pd

from fetch_ipl_balls import BALLS_DIR, COVERAGE_CSV, all_ball_files, scrape
from fetch_ipl_matches import update_index

# An IPL week is at most ~10 matches; this leaves headroom for stale re-fetches.
DEFAULT_LIMIT = 40

# 'Scorecard only' is the one coverage value that reliably means no ball data.
NO_BALL_DATA = "Scorecard only"


def load_coverage():
    """gid -> match_status recorded at scrape time."""
    if not os.path.exists(COVERAGE_CSV):
        return {}
    cov = pd.read_csv(COVERAGE_CSV, dtype=str, keep_default_na=False)
    return dict(zip(cov["gid"], cov["match_status"]))


def pending(matches_df, from_season, limit=None):
    """Matches needing a ball fetch, newest first, split by reason."""
    df = matches_df[matches_df["source_status"] == "results"]
    if from_season:
        years = pd.to_numeric(df["ipl_season"], errors="coerce")
        df = df[years >= from_season]
    df = df[~df["coverage_detail"].str.startswith(NO_BALL_DATA, na=False)]

    have = {os.path.basename(f)[:-4] for f in all_ball_files()}
    coverage = load_coverage()

    missing, stale = [], []
    for gid in df.sort_values("start_date", ascending=False)["gid"]:
        if gid not in have:
            missing.append(gid)
        elif coverage.get(gid, "") != "complete":
            stale.append(gid)

    gids = missing + stale
    if limit:
        gids = gids[:limit]
    return gids, len(missing), len(stale)


def summarise():
    files = all_ball_files() if os.path.isdir(BALLS_DIR) else []
    if not files:
        return
    balls = sum(len(pd.read_csv(f, usecols=["scoring"], low_memory=False)) for f in files)
    print(f"\ncorpus: {len(files)} matches, {balls} deliveries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", help="seasons to refresh (default: current)")
    parser.add_argument("--all-seasons", action="store_true", help="rebuild the whole index")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"max matches to fetch this run (default {DEFAULT_LIMIT}; 0 = no cap)")
    parser.add_argument("--from-season", type=int, default=None,
                        help="earliest season to fetch balls for (default: current year; 0 = all)")
    parser.add_argument("--raw", action="store_true", help="keep raw JSON")
    args = parser.parse_args()

    from_season = time.gmtime().tm_year if args.from_season is None else args.from_season

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
    gids, n_missing, n_stale = pending(matches_df, from_season, limit=args.limit or None)
    print(f"scope: season >= {from_season or 'all'}")
    print(f"{n_missing} never fetched, {n_stale} incomplete when last fetched")

    if not gids:
        print("nothing to fetch -- corpus is up to date")
    else:
        if args.limit and n_missing + n_stale > args.limit:
            print(f"capped at {args.limit} this run; rerun to continue")
        scrape(gids, raw=args.raw, refresh=True)

    print("\n" + "=" * 60)
    print("STEP 3  summary")
    print("=" * 60)
    summarise()


if __name__ == "__main__":
    main()
