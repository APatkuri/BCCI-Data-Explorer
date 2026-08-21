"""Assemble one ball-level table per match from the ellipsedata endpoints.

Commentary is the spine: it carries every delivery, including ones the ball
tracking missed. Everything else left-joins onto it by
``(innings_number, overs_unique)``.

Tracking coverage is not implied by the match index's ``coverage_detail`` --
plenty of matches report "BBB all details" yet return an empty ``tracking_data``
(this is common for older and women's fixtures). So per-match coverage is
recorded in ``data/ball_coverage.csv`` rather than assumed.

Usage:
    python fetch_balls.py --gid 9492698a-0cbe-4faa-ab47-44c1048dc8d0
    python fetch_balls.py --limit 25             # from matches.csv, newest first
    python fetch_balls.py --season 2026 --class international
    python fetch_balls.py --limit 50 --raw       # also keep the raw JSON
"""

import argparse
import glob
import json
import os

import pandas as pd

from commentary_parser import enrich
from ellipse_api import fetch_match, make_session, spider_key
from fetch_matches import classify

try:
    from tqdm import tqdm
except ImportError:  # optional -- fall back to plain numbered lines
    tqdm = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BALLS_DIR = os.path.join(DATA_DIR, "balls")
RAW_DIR = os.path.join(DATA_DIR, "raw", "ellipse")
MATCHES_CSV = os.path.join(DATA_DIR, "matches.csv")
COVERAGE_CSV = os.path.join(DATA_DIR, "ball_coverage.csv")

# Player/bowler descriptors repeated verbatim across pitchmap, beehive and
# wagon. Taken once from whichever tracking source is present.
SHARED_TRACKING = (
    "batting_player_id", "batting_player_name", "batting_player_hand",
    "bowling_player_id", "bowling_player_name", "bowling_player_hand",
    "batting_team_id", "batting_team_name", "bowling_team_id", "bowling_team_name",
    "bowling_technique", "bowling_type_id", "bowling_type_simple", "datetime_utc",
)

PITCHMAP_ONLY = ("bounce_position_x", "bounce_position_y", "length_zone")
BEEHIVE_ONLY = ("stumps_position_y", "stumps_position_z", "grid_x", "grid_y")
SPIDER_ONLY = (
    "ball_id", "field_direction", "field_distance_percent",
    "field_magnitude", "field_zone", "runs_off_bat", "runs_conceded",
)


def load_class_map():
    """gid -> class_names from the index, for foldering ball tables."""
    if not os.path.exists(MATCHES_CSV):
        return {}
    df = pd.read_csv(MATCHES_CSV, dtype=str, keep_default_na=False,
                     usecols=["gid", "class_names"])
    return dict(zip(df["gid"], df["class_names"]))


def ball_path(gid, class_names):
    """data/balls/{format}/{gender}_{level}/{gid}.csv

    Format folders make the corpus browsable by the split people actually care
    about -- Test/ODI/T20, senior/youth, men/women -- instead of 2800 uuids in
    one directory.
    """
    match_format, gender, level = classify(class_names)
    return os.path.join(BALLS_DIR, match_format, f"{gender}_{level}", f"{gid}.csv")


def find_existing(gid):
    """Locate a ball table wherever it currently sits, foldered or not."""
    hits = glob.glob(os.path.join(BALLS_DIR, "**", f"{gid}.csv"), recursive=True)
    return hits[0] if hits else None


def all_ball_files():
    return sorted(glob.glob(os.path.join(BALLS_DIR, "**", "*.csv"), recursive=True))


def index_tracking(payload):
    """``{(innings, overs_unique): record}`` from a pitchmap/beehive payload."""
    if not payload:
        return {}
    return {
        (r["innings_number"], r["overs_unique"]): r
        for r in payload.get("tracking_data") or []
    }


def index_wagon(payload, section):
    if not payload:
        return {}
    return {
        (r["innings_number"], spider_key(r)): r
        for r in payload.get(section) or []
    }


def commentary_rows(payload):
    """Flatten ``innings[].bbb[].balls[]`` into one row per delivery.

    Both the innings list and the per-over ball list arrive newest-first, so the
    result is sorted into playing order at the end.
    """
    if not payload:
        return []

    rows = []
    for innings in payload.get("innings") or []:
        innings_no = innings.get("innings_number")
        innings_meta = {
            "innings_number": innings_no,
            "batting_team_id": innings.get("batting_team_id"),
            "batting_team_name": innings.get("batting_team_name"),
            "batting_team_abbreviation": innings.get("batting_team_abbreviation"),
            "innings_closure_id": innings.get("closure_id"),
        }

        for over in innings.get("bbb") or []:
            over_meta = {
                "over_number": over.get("over_number"),
                "over_bowler_name": over.get("bowling_player_name"),
                "runs_off_over": over.get("runs_off_over"),
                "extras_off_over": over.get("extras_off_over"),
                "wickets_off_over": over.get("wickets_off_over"),
                "team_score": over.get("team_score"),
                "run_rate": over.get("run_rate"),
            }

            for ball in over.get("balls") or []:
                commentary = ball.get("commentary") or {}
                rows.append({
                    **innings_meta,
                    **over_meta,
                    "overs_unique": ball.get("overs_unique"),
                    "display_overs": ball.get("display_overs"),
                    "runs": ball.get("runs"),
                    "scoring": ball.get("scoring"),
                    "ball_summary_text": commentary.get("ball_summary_text"),
                    "ball_pre_text": commentary.get("ball_pre_text"),
                    "commentary": commentary.get("message"),
                })

    rows.sort(key=lambda r: (r["innings_number"], _over_sort_key(r["overs_unique"])))
    return rows


def _over_sort_key(overs_unique):
    """'77.04' -> (77, 4) so overs sort numerically rather than as text."""
    try:
        over, ball = str(overs_unique).split(".")
        return int(over), int(ball)
    except (ValueError, AttributeError):
        return (0, 0)


def build_match_table(gid, payloads):
    """Join every endpoint onto the commentary spine. Returns (DataFrame, coverage)."""
    commentary = payloads.get("commentary")
    rows = commentary_rows(commentary)

    pitchmap = index_tracking(payloads.get("pitchmap"))
    beehive = index_tracking(payloads.get("beehive"))
    spider = index_wagon(payloads.get("wagon"), "spider_data")
    catches = index_wagon(payloads.get("wagon"), "catch_map")

    summary = payloads.get("summary") or {}
    info = (payloads.get("pitchmap") or payloads.get("beehive") or {}).get("info") or {}

    match_meta = {
        "gid": gid,
        "match_id": info.get("id") or summary.get("event_id"),
        "match_name": info.get("match_name") or summary.get("title"),
        "comp_id": info.get("comp_id"),
        "comp_name": info.get("comp_name") or summary.get("comp_name"),
        "comp_season": summary.get("comp_season"),
        "ground_name": info.get("ground_name") or summary.get("ground_name"),
        "class_name": (commentary or {}).get("class_name"),
        "start_date": info.get("start_date") or summary.get("start_date"),
    }

    for row in rows:
        key = (row["innings_number"], row["overs_unique"])
        pm, bh, sp = pitchmap.get(key), beehive.get(key), spider.get(key)

        row.update(match_meta)

        # Player descriptors: identical wherever they appear, so first hit wins.
        source = pm or bh or sp or {}
        for field in SHARED_TRACKING:
            row[field] = row.get(field) or source.get(field)

        for field in PITCHMAP_ONLY:
            row[field] = (pm or {}).get(field)
        for field in BEEHIVE_ONLY:
            row[field] = (bh or {}).get(field)
        for field in SPIDER_ONLY:
            row[field] = (sp or {}).get(field)

        row["has_pitchmap"] = pm is not None
        row["has_beehive"] = bh is not None
        row["has_wagon"] = sp is not None
        row["in_catch_map"] = key in catches

    df = pd.DataFrame(rows)

    coverage = {
        "gid": gid,
        "match_id": match_meta["match_id"],
        "match_name": match_meta["match_name"],
        "class_name": match_meta["class_name"],
        "start_date": match_meta["start_date"],
        # Status at scrape time. A match scraped mid-innings has partial ball
        # data, so the weekly update re-fetches anything not 'complete'.
        "match_status": (commentary or {}).get("match_status"),
        "balls": len(df),
        "pitchmap_balls": len(pitchmap),
        "beehive_balls": len(beehive),
        "wagon_balls": len(spider),
        "catch_map_balls": len(catches),
        "has_tracking": bool(pitchmap or beehive),
    }
    return df, coverage


def write_raw(gid, payloads):
    target = os.path.join(RAW_DIR, gid)
    os.makedirs(target, exist_ok=True)
    for name, payload in payloads.items():
        if payload is None:
            continue
        with open(os.path.join(target, f"{name}.json"), "w") as handle:
            json.dump(payload, handle)


def select_gids(args):
    """Which matches to scrape -- an explicit gid, or a slice of matches.csv."""
    if args.gid:
        return list(args.gid)

    if not os.path.exists(MATCHES_CSV):
        raise SystemExit(f"{MATCHES_CSV} not found -- run fetch_matches.py first")

    df = pd.read_csv(MATCHES_CSV, dtype=str, keep_default_na=False)
    df = df[df["source_status"] == "results"]

    if args.season:
        df = df[df["season"].isin(args.season)]
    if args.match_class:
        df = df[df["match_class"] == args.match_class]
    if not args.include_untracked:
        df = df[df["has_bbb"] == "True"]

    df = df.sort_values("start_date", ascending=False)
    if args.limit:
        df = df.head(args.limit)
    return df["gid"].tolist()


def scrape(gids, raw=False, refresh=False):
    """Fetch, join, parse and write a ball table per gid. Returns coverage rows.

    Importable so the weekly update can drive it without shelling out.
    """
    os.makedirs(BALLS_DIR, exist_ok=True)
    session = make_session()
    class_map = load_class_map()
    coverage_rows = []
    print(f"{len(gids)} match(es) to fetch\n")

    # tqdm owns the last terminal line, so per-match detail has to go through
    # tqdm.write or it gets overwritten by the bar.
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

        # Parsed inline so each match has exactly one table on disk. Re-parsing
        # later is parse_balls.py's job, and it rewrites these same files.
        df = enrich(df)

        # Index class is richer than the commentary's own label, so prefer it.
        class_names = class_map.get(gid) or coverage.get("class_name") or ""
        out_path = ball_path(gid, class_names)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # A reclassified match must not leave a copy in its old folder.
        if existing and os.path.abspath(existing) != os.path.abspath(out_path):
            os.remove(existing)

        df.to_csv(out_path, index=False)
        coverage["path"] = os.path.relpath(out_path, BALLS_DIR)
        coverage_rows.append(coverage)
        if raw:
            write_raw(gid, payloads)

        totals["balls"] += coverage["balls"]
        totals["tracked"] += 1 if coverage["has_tracking"] else 0

        track = (
            f"track {coverage['pitchmap_balls']}"
            if coverage["has_tracking"] else "NO tracking"
        )
        log(
            f"  {gid[:8]} {coverage['balls']:5d} balls | {track:12s} "
            f"| wagon {coverage['wagon_balls']}"
        )
        if bar:
            bar.set_postfix(balls=totals["balls"], tracked=totals["tracked"],
                            skipped=totals["skipped"], refresh=False)

    if bar:
        bar.close()

    write_coverage(coverage_rows)
    print(f"ball tables -> {BALLS_DIR}")
    return coverage_rows


def write_coverage(coverage_rows):
    if not coverage_rows:
        return
    cov = pd.DataFrame(coverage_rows)
    if os.path.exists(COVERAGE_CSV):
        old = pd.read_csv(COVERAGE_CSV, dtype=str, keep_default_na=False)
        cov = pd.concat([old, cov.astype(str)], ignore_index=True)
        cov = cov.drop_duplicates(subset=["gid"], keep="last")
    cov.to_csv(COVERAGE_CSV, index=False)
    print(f"\ncoverage -> {COVERAGE_CSV}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gid", nargs="+", help="specific match gid(s)")
    parser.add_argument("--season", nargs="+", help="filter matches.csv by season")
    parser.add_argument("--class", dest="match_class", choices=("international", "domestic"))
    parser.add_argument("--limit", type=int, help="cap number of matches")
    parser.add_argument("--include-untracked", action="store_true",
                        help="also scrape matches whose index says no ball-by-ball")
    parser.add_argument("--raw", action="store_true", help="keep raw JSON per match")
    parser.add_argument("--refresh", action="store_true", help="re-scrape matches already on disk")
    args = parser.parse_args()

    scrape(select_gids(args), raw=args.raw, refresh=args.refresh)


if __name__ == "__main__":
    main()
