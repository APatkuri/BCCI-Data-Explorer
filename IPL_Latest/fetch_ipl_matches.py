"""Build the IPL match index: fixtures + results for every season.

    iplt20.com BFF (season)  ->  competition externalGid
                             ->  stats.bcci.tv/match/{fixtures,results}/?comp_gid=...
                             ->  data/matches.csv + data/match_innings.csv

Rows are flattened with BCCI_Latest's ``normalise_match``, so the columns match
``BCCI_Latest/data/matches.csv`` and the two indexes can be concatenated.
``match_class`` is ``ipl``.

Usage:
    python fetch_ipl_matches.py                  # current season
    python fetch_ipl_matches.py --season 2025 2024
    python fetch_ipl_matches.py --all-seasons    # full backfill to 2008
    python fetch_ipl_matches.py --raw            # keep the source JSON too
"""

import argparse
import json
import os
import time

import pandas as pd

from ipl_api import available_seasons, competition_gids, default_season, fetch_filter_config
from ipl_api import make_session as make_bff_session
from shared import fetch_matches, innings_rows, make_stats_session, merge_existing, normalise_match

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "matches")

MATCH_CLASS = "ipl"
STATUSES = ("results", "fixtures")


def resolve_seasons(bff, seasons=None, all_seasons=False):
    if seasons:
        return list(seasons)
    config = fetch_filter_config(bff)
    if not config:
        raise SystemExit("iplt20.com BFF returned nothing -- cannot resolve seasons")
    if all_seasons:
        return available_seasons(config)
    return [default_season(config) or str(time.gmtime().tm_year)]


def collect(bff, stats, seasons, statuses):
    all_matches, all_innings, raw_by_key = [], [], {}

    for season in seasons:
        config = fetch_filter_config(bff, season)
        gids = competition_gids(config) if config else []
        if not gids:
            print(f"  {season}: no competitions, skipping")
            continue

        for status in statuses:
            matches = fetch_matches(stats, status, gids)
            print(f"  {season} {status:8s}: {len(matches):4d} matches")

            raw_by_key[(season, status)] = matches
            for match in matches:
                row = normalise_match(match, MATCH_CLASS, status)
                # stats.bcci.tv's own ``season`` is unreliable for the IPL --
                # 2010 comes back as '2009/10', 2020 as '2020/21', and part of
                # 2022 as '2021/22'. The BFF year the gid was looked up under is
                # the edition the match belongs to.
                row["ipl_season"] = season
                all_matches.append(row)
                all_innings.extend(innings_rows(match))

            time.sleep(0.5)

    return all_matches, all_innings, raw_by_key


def write_raw(raw_by_key):
    os.makedirs(RAW_DIR, exist_ok=True)
    for (season, status), matches in raw_by_key.items():
        with open(os.path.join(RAW_DIR, f"ipl_{season}_{status}.json"), "w") as handle:
            json.dump(matches, handle, indent=2)


def update_index(seasons=None, all_seasons=False, statuses=STATUSES, merge=True, raw=False):
    """Refresh matches.csv / match_innings.csv. Returns the match DataFrame.

    Importable so the weekly update can drive it without shelling out.
    """
    bff, stats = make_bff_session(), make_stats_session()
    os.makedirs(DATA_DIR, exist_ok=True)

    seasons = resolve_seasons(bff, seasons, all_seasons)
    print(f"ipl: {len(seasons)} season(s)")
    matches, innings, raw_payloads = collect(bff, stats, seasons, statuses)

    if raw:
        write_raw(raw_payloads)

    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        # A match can appear under both statuses mid-transition; results wins
        # because STATUSES lists it first and keep='first' holds on to it.
        matches_df = matches_df.drop_duplicates(subset=["match_id"], keep="first")
    innings_df = pd.DataFrame(innings)
    if not innings_df.empty:
        innings_df = innings_df.drop_duplicates(subset=["match_id", "innings_number"], keep="last")

    matches_path = os.path.join(DATA_DIR, "matches.csv")
    innings_path = os.path.join(DATA_DIR, "match_innings.csv")

    if merge:
        matches_df = merge_existing(matches_df, matches_path, ["match_id"])
        innings_df = merge_existing(innings_df, innings_path, ["match_id", "innings_number"])

    if not matches_df.empty and "start_date" in matches_df.columns:
        matches_df = matches_df.sort_values("start_date")

    matches_df.to_csv(matches_path, index=False)
    if not innings_df.empty:
        innings_df.to_csv(innings_path, index=False)

    print(
        f"\n{len(matches_df)} matches -> {matches_path}"
        f"\n{len(innings_df)} innings -> {innings_path}"
    )
    return matches_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", help="season years, e.g. 2026 2025")
    parser.add_argument("--all-seasons", action="store_true", help="every season the BFF lists")
    parser.add_argument("--status", nargs="+", choices=STATUSES, default=list(STATUSES))
    parser.add_argument("--no-merge", action="store_true", help="overwrite CSVs instead of merging")
    parser.add_argument("--raw", action="store_true", help="also keep the raw JSON per season/status")
    args = parser.parse_args()

    update_index(
        seasons=args.season,
        all_seasons=args.all_seasons,
        statuses=args.status,
        merge=not args.no_merge,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()
