"""One ball-level table per IPL match, from the ellipsedata endpoints.

Fetching, joining and parsing are BCCI_Latest's (``fetch_match``,
``build_match_table``, ``enrich``), so the 85 columns are identical to
``BCCI_Latest/data/balls``. What differs is layout: every IPL match is a men's
T20, so the format/category folders BCCI_Latest uses would all collapse into one
-- tables are foldered by season instead.

IPL tracking is thin: spot checks across 2009, 2018 and 2025 returned wagon
data but no pitchmap/beehive at all. ``data/ball_coverage.csv`` records what
each match actually returned.

Usage:
    python fetch_ipl_balls.py --gid b5c11c68-16e6-47dc-9f24-953b8985a763
    python fetch_ipl_balls.py --season 2025          # from matches.csv
    python fetch_ipl_balls.py --limit 25             # newest first
    python fetch_ipl_balls.py --refresh --season 2026
"""

import argparse
import glob
import json
import os

import pandas as pd

from shared import build_match_table, enrich, fetch_match, make_ellipse_session

try:
    from tqdm import tqdm
except ImportError:  # optional -- fall back to plain numbered lines
    tqdm = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BALLS_DIR = os.path.join(DATA_DIR, "balls")
RAW_DIR = os.path.join(DATA_DIR, "raw", "ellipse")
MATCHES_CSV = os.path.join(DATA_DIR, "matches.csv")
COVERAGE_CSV = os.path.join(DATA_DIR, "ball_coverage.csv")


def load_season_map():
    """gid -> IPL edition year from the index, for foldering ball tables."""
    if not os.path.exists(MATCHES_CSV):
        return {}
    df = pd.read_csv(MATCHES_CSV, dtype=str, keep_default_na=False, usecols=["gid", "ipl_season"])
    return dict(zip(df["gid"], df["ipl_season"]))


def ball_path(gid, season):
    """data/balls/{season}/{gid}.csv"""
    return os.path.join(BALLS_DIR, season or "unknown", f"{gid}.csv")


def find_existing(gid):
    hits = glob.glob(os.path.join(BALLS_DIR, "**", f"{gid}.csv"), recursive=True)
    return hits[0] if hits else None


def all_ball_files():
    return sorted(glob.glob(os.path.join(BALLS_DIR, "**", "*.csv"), recursive=True))


def write_raw(gid, payloads):
    target = os.path.join(RAW_DIR, gid)
    os.makedirs(target, exist_ok=True)
    for name, payload in payloads.items():
        if payload is not None:
            with open(os.path.join(target, f"{name}.json"), "w") as handle:
                json.dump(payload, handle)


def select_gids(args):
    """Which matches to scrape -- explicit gids, or a slice of matches.csv."""
    if args.gid:
        return list(args.gid)

    if not os.path.exists(MATCHES_CSV):
        raise SystemExit(f"{MATCHES_CSV} not found -- run fetch_ipl_matches.py first")

    df = pd.read_csv(MATCHES_CSV, dtype=str, keep_default_na=False)
    df = df[df["source_status"] == "results"]
    if args.season:
        df = df[df["ipl_season"].isin(args.season)]

    df = df.sort_values("start_date", ascending=False)
    if args.limit:
        df = df.head(args.limit)
    return df["gid"].tolist()


def scrape(gids, raw=False, refresh=False):
    """Fetch, join, parse and write a ball table per gid. Returns coverage rows.

    Coverage is written after every match rather than once at the end, so a
    long backfill that dies partway still leaves an accurate ball_coverage.csv
    for everything already on disk.
    """
    os.makedirs(BALLS_DIR, exist_ok=True)
    session = make_ellipse_session()
    season_map = load_season_map()
    coverage_rows = []
    print(f"{len(gids)} match(es) to fetch\n")

    bar = tqdm(gids, unit="match", desc="fetching") if tqdm else None
    log = tqdm.write if tqdm else print
    totals = {"balls": 0, "tracked": 0, "skipped": 0}

    for gid in bar or gids:
        existing = find_existing(gid)
        if existing and not refresh:
            totals["skipped"] += 1
            log(f"  {gid[:8]} cached, skipping")
            continue

        payloads = fetch_match(session, gid)
        if not payloads.get("commentary"):
            totals["skipped"] += 1
            log(f"  {gid[:8]} no commentary, skipped")
            continue

        df, coverage = build_match_table(gid, payloads)
        if df.empty:
            totals["skipped"] += 1
            log(f"  {gid[:8]} no balls, skipped")
            continue

        df = enrich(df)

        season = season_map.get(gid) or (payloads.get("summary") or {}).get("comp_season")
        out_path = ball_path(gid, season)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if existing and os.path.abspath(existing) != os.path.abspath(out_path):
            os.remove(existing)

        df.to_csv(out_path, index=False)
        coverage["season"] = season
        coverage["path"] = os.path.relpath(out_path, BALLS_DIR)
        coverage_rows.append(coverage)
        write_coverage([coverage])
        if raw:
            write_raw(gid, payloads)

        totals["balls"] += coverage["balls"]
        totals["tracked"] += 1 if coverage["has_tracking"] else 0
        track = f"track {coverage['pitchmap_balls']}" if coverage["has_tracking"] else "NO tracking"
        log(f"  {gid[:8]} {coverage['balls']:5d} balls | {track:12s} | wagon {coverage['wagon_balls']}")
        if bar:
            bar.set_postfix(balls=totals["balls"], tracked=totals["tracked"],
                            skipped=totals["skipped"], refresh=False)

    if bar:
        bar.close()

    if coverage_rows:
        print(f"\ncoverage -> {COVERAGE_CSV}")
    print(f"ball tables -> {BALLS_DIR}")
    return coverage_rows


def write_coverage(coverage_rows):
    cov = pd.DataFrame(coverage_rows).astype(str)
    if os.path.exists(COVERAGE_CSV):
        old = pd.read_csv(COVERAGE_CSV, dtype=str, keep_default_na=False)
        cov = pd.concat([old, cov], ignore_index=True).drop_duplicates(subset=["gid"], keep="last")
    cov.sort_values("start_date").to_csv(COVERAGE_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gid", nargs="+", help="specific match gid(s)")
    parser.add_argument("--season", nargs="+", help="filter matches.csv by season")
    parser.add_argument("--limit", type=int, help="cap number of matches")
    parser.add_argument("--raw", action="store_true", help="keep raw JSON per match")
    parser.add_argument("--refresh", action="store_true", help="re-scrape matches already on disk")
    args = parser.parse_args()

    scrape(select_gids(args), raw=args.raw, refresh=args.refresh)


if __name__ == "__main__":
    main()
