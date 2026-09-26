"""Re-parse the IPL ball tables in place and report what the parser recovered.

Only needed after ``BCCI_Latest/commentary_parser.py`` changes -- fetch_ipl_balls.py
already parses as it writes.

Usage:
    python parse_ipl_balls.py                # re-parse in place, print yield report
    python parse_ipl_balls.py --report-only  # report on what's on disk, write nothing
    python parse_ipl_balls.py --combined     # additionally export one concatenated file
"""

import argparse
import os

import pandas as pd

from fetch_ipl_balls import BALLS_DIR, DATA_DIR, all_ball_files
from shared import enrich, yield_report

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--combined", action="store_true",
                        help="also write data/balls_combined.csv (a derived export)")
    args = parser.parse_args()

    files = all_ball_files()
    if not files:
        raise SystemExit(f"no ball tables in {BALLS_DIR} -- run fetch_ipl_balls.py first")

    frames = []
    for path in tqdm(files, unit="match", desc="parsing") if tqdm else files:
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


if __name__ == "__main__":
    main()
