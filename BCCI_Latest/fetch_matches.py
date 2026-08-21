"""Build the match index: fixtures + results for every class/season.

Pipeline, mirroring how the site itself queries:

    BFF (class, season)  ->  competition externalGids
                         ->  stats.bcci.tv/match/{fixtures,results}/?comp_gid=...
                         ->  data/matches.csv + data/match_innings.csv

The resulting ``match_id``/``gid`` pairs are the entry point for ball-by-ball
scraping later; ``has_bbb`` marks which matches actually carry it.

Usage:
    python fetch_matches.py                      # current season, both classes
    python fetch_matches.py --season 2025 2024
    python fetch_matches.py --all-seasons        # full backfill to 2012
    python fetch_matches.py --class international
    python fetch_matches.py --raw                # keep the source JSON too
"""

import argparse
import json
import os
import re
import time

import pandas as pd

from bcci_api import (
    available_seasons,
    competition_gids,
    fetch_filter_config,
    fetch_matches,
    make_session,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "matches")

CLASSES = ("international", "domestic")
STATUSES = ("results", "fixtures")

# Nested/bulky fields handled separately (or dropped) rather than flattened.
NESTED_FIELDS = ("class", "summary", "weather")

# The index reports class as a pipe-joined list -- 'Tests|First-class',
# "Women's Twenty20 internationals|Women's competitive Twenty20". Three
# orthogonal things are baked into that string, so they get pulled apart into
# separate columns rather than left as one opaque label.
FORMAT_RULES = (
    (r"twenty20|t20", "T20"),
    (r"one-day|odi|list a", "ODI"),
    (r"test|first-class|two-innings|multi-day", "Test"),
)


def season_end_year(season):
    """'2025/26' -> 2026, '2026' -> 2026, '2026/27' -> 2027.

    Seasons arrive in two spellings -- a bare year for tournaments inside one
    calendar year, and a split year for those straddling two. Reducing both to
    the year the season *ends* gives a single sortable key, so "2026 onwards"
    means one comparison instead of a string match per spelling.
    """
    text = str(season).strip()
    if re.fullmatch(r"\d{4}", text):
        return int(text)
    split = re.fullmatch(r"(\d{4})/(\d{2})", text)
    if split:
        return int(split.group(1)[:2] + split.group(2))
    return None


def classify(class_names):
    """'Tests|First-class' -> ('Test', 'men', 'international').

    Order matters: Twenty20 is checked before one-day because 'List A Twenty20'
    contains both, and 'List A' alone means a 50-over domestic game.
    """
    text = (class_names or "").lower()

    match_format = next((f for pattern, f in FORMAT_RULES if re.search(pattern, text)), "Other")
    gender = "women" if "women" in text else "men"

    if "youth" in text or "under-19" in text or "u19" in text:
        level = "youth"
    elif "international" in text or re.search(r"\btests\b", text):
        # 'Tests' with no qualifier is the international competition; 'Test' as
        # a format word shows up in domestic strings too, hence the word bound.
        level = "international"
    else:
        level = "domestic"

    return match_format, gender, level


def normalise_match(match, match_class, status):
    """One match dict -> one flat row."""
    row = {k: v for k, v in match.items() if k not in NESTED_FIELDS}

    row["match_class"] = match_class
    row["source_status"] = status

    classes = match.get("class") or []
    row["class_names"] = "|".join(c.get("name", "") for c in classes)
    row["class_ids"] = "|".join(str(c.get("id", "")) for c in classes)

    # 'Scorecard, BBB all details (live)' vs 'Scorecard only (post-match)'.
    coverage = match.get("coverage_detail") or ""
    row["has_bbb"] = "BBB" in coverage
    row["bbb_reduced"] = "BBB reduced" in coverage

    match_format, gender, level = classify(row["class_names"])
    row["format"] = match_format
    row["gender"] = gender
    row["level"] = level
    row["category"] = f"{gender}_{level}"
    row["season_end_year"] = season_end_year(row.get("season"))

    summary = match.get("summary") or {}
    row["innings_count"] = len(summary.get("innings") or [])

    return row


def innings_rows(match):
    """Per-innings score lines from the embedded summary."""
    summary = match.get("summary") or {}
    rows = []
    for innings in summary.get("innings") or []:
        row = dict(innings)
        row["match_id"] = match.get("match_id")
        row["gid"] = match.get("gid")
        rows.append(row)
    return rows


def collect(session, match_class, seasons, statuses):
    all_matches, all_innings, raw_by_key = [], [], {}

    for season in seasons:
        config = fetch_filter_config(session, match_class, season)
        if not config:
            print(f"  {match_class} {season}: no filter config, skipping")
            continue

        gids = competition_gids(config)
        if not gids:
            print(f"  {match_class} {season}: no competitions")
            continue

        for status in statuses:
            matches = fetch_matches(session, status, gids)
            print(
                f"  {match_class} {season} {status:8s}: "
                f"{len(gids):3d} comps -> {len(matches):5d} matches"
            )

            raw_by_key[(match_class, season, status)] = matches
            for match in matches:
                all_matches.append(normalise_match(match, match_class, status))
                all_innings.extend(innings_rows(match))

            time.sleep(0.5)

    return all_matches, all_innings, raw_by_key


def write_raw(raw_by_key):
    os.makedirs(RAW_DIR, exist_ok=True)
    for (match_class, season, status), matches in raw_by_key.items():
        path = os.path.join(RAW_DIR, f"{match_class}_{season}_{status}.json")
        with open(path, "w") as handle:
            json.dump(matches, handle, indent=2)


def merge_existing(df, path, subset):
    """Fold new rows into whatever is already on disk, new rows winning."""
    if not os.path.exists(path) or df.empty:
        return df

    existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    if existing.empty:
        return df

    combined = pd.concat([existing, df.astype(str)], ignore_index=True)
    return combined.drop_duplicates(subset=subset, keep="last")


def update_index(classes=CLASSES, seasons=None, all_seasons=False,
                 statuses=STATUSES, merge=True, raw=False):
    """Refresh matches.csv / match_innings.csv. Returns the match DataFrame.

    ``seasons=None`` with ``all_seasons=False`` means the BFF's current default
    season -- the right choice for a scheduled update. Importable so the weekly
    update can drive it without shelling out.
    """
    session = make_session()
    os.makedirs(DATA_DIR, exist_ok=True)

    matches, innings, raw_payloads = [], [], {}

    for match_class in classes:
        if all_seasons:
            config = fetch_filter_config(session, match_class)
            class_seasons = available_seasons(config) if config else []
        elif seasons:
            class_seasons = seasons
        else:
            config = fetch_filter_config(session, match_class)
            filters = (config or {}).get("filters") or {}
            class_seasons = [filters.get("defaultSeasonYear", "2026")]

        print(f"{match_class}: {len(class_seasons)} season(s)")
        cls_matches, cls_innings, cls_raw = collect(
            session, match_class, class_seasons, statuses
        )
        matches.extend(cls_matches)
        innings.extend(cls_innings)
        raw_payloads.update(cls_raw)

    if raw:
        write_raw(raw_payloads)

    matches_df = pd.DataFrame(matches).drop_duplicates(subset=["match_id"], keep="last")
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

    bbb = matches_df[matches_df["has_bbb"].astype(str) == "True"] if "has_bbb" in matches_df else []
    print(
        f"\n{len(matches_df)} matches -> {matches_path}"
        f"\n{len(innings_df)} innings -> {innings_path}"
        f"\n{len(bbb)} matches carry ball-by-ball"
    )
    return matches_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", nargs="+", help="season years, e.g. 2026 2025")
    parser.add_argument("--all-seasons", action="store_true", help="every season the BFF lists")
    parser.add_argument("--class", dest="classes", nargs="+", choices=CLASSES, default=list(CLASSES))
    parser.add_argument("--status", nargs="+", choices=STATUSES, default=list(STATUSES))
    parser.add_argument("--no-merge", action="store_true", help="overwrite CSVs instead of merging")
    parser.add_argument("--raw", action="store_true",
                        help="also keep the raw JSON per class/season/status")
    args = parser.parse_args()

    update_index(
        classes=args.classes,
        seasons=args.season,
        all_seasons=args.all_seasons,
        statuses=args.status,
        merge=not args.no_merge,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()
