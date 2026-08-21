"""Re-parse the ball tables in place and report what the parser recovered.

fetch_balls.py already parses each match as it writes it, so this is only
needed when the vocabularies in commentary_parser.py change and the existing
tables should pick up the improvement. It rewrites ``data/balls/*.csv`` --
there is no second copy of the data anywhere.

Usage:
    python parse_balls.py                # re-parse in place, print yield report
    python parse_balls.py --report-only  # report on what's on disk, write nothing
    python parse_balls.py --combined     # additionally export one concatenated file
"""

import argparse
import glob
import os

import pandas as pd

from commentary_parser import PARSED_COLS, SCORING_COLS, enrich

try:
    from tqdm import tqdm
except ImportError:  # optional -- the loop just runs without a bar
    tqdm = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BALLS_DIR = os.path.join(DATA_DIR, "balls")


def yield_report(df):
    print(f"\n{'field':20s} {'filled':>8s} {'pct':>7s}")
    print("-" * 38)
    for col in PARSED_COLS[:-1] + SCORING_COLS[:2]:
        filled = df[col].notna().sum()
        print(f"{col:20s} {filled:8d} {100 * filled / len(df):6.1f}%")

    print(f"\n{'parse_method':26s} {'balls':>7s} {'pct':>7s}")
    print("-" * 42)
    for method, count in df["parse_method"].value_counts().items():
        print(f"{method:26s} {count:7d} {100 * count / len(df):6.1f}%")

    print("\nyield by parse_method (share of rows with a value):")
    fields = ["length", "line", "footwork", "shot_type", "field_position"]
    print(df.groupby("parse_method")[fields].apply(lambda g: g.notna().mean().round(2)).to_string())

    wickets = df[df["is_wicket"].eq(True)]
    if len(wickets):
        found = wickets["dismissal_type"].notna().sum()
        false_pos = df[~df["is_wicket"].eq(True)]["dismissal_type"].notna().sum()
        print(
            f"\nwickets {len(wickets)} | mode resolved {found} "
            f"({100 * found / len(wickets):.0f}%) | false positives {false_pos}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--combined", action="store_true",
                        help="also write data/balls_combined.csv (a derived export)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(BALLS_DIR, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"no ball tables in {BALLS_DIR} -- run fetch_balls.py first")

    frames = []
    paths = tqdm(files, unit="match", desc="parsing") if tqdm else files
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        if not args.report_only:
            df = enrich(df)
            df.to_csv(path, index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    action = "read" if args.report_only else "re-parsed"
    print(f"{len(files)} match(es) {action}, {len(combined)} deliveries")
    yield_report(combined)

    if args.combined:
        out = os.path.join(DATA_DIR, "balls_combined.csv")
        combined.to_csv(out, index=False)
        print(f"\ncombined export -> {out}")
    elif not args.report_only:
        print(f"\nrewritten in place -> {BALLS_DIR}")


if __name__ == "__main__":
    main()
