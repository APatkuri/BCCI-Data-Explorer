import math
import socket
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_scrapper import fetch_bbb_data
from process_data import processData


def check_connectivity(host='polls.iplt20.com', port=443, timeout=5):
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((host, port))
        return True
    except OSError:
        return False

FIELDS = [
    'MatchID', 'InningsNo', 'BattingTeamID', 'TeamName', 'BatsManName', 'BowlerName', 'BowlerType', 'OverNo',
    'BallNo', 'Runs', 'BallRuns', 'ActualRuns', 'IsOne', 'IsTwo', 'IsThree', 'IsDotball', 'Extras', 'IsWide',
    'IsNoBall', 'IsBye', 'IsLegBye', 'IsFour', 'IsSix', 'IsWicket', 'WicketType', 'Wickets', 'IsBowlerWicket',
    'BallName', 'Day', 'SESSION_NO', 'IsExtra', 'SNO', 'Xpitch', 'Ypitch', 'RunRuns', 'IsMaiden', 'OverImage',
    'BowlTypeID', 'BowlTypeName', 'ShotTypeID', 'ShotType', 'IsBouncer', 'IsFreeHit', 'BallCount', 'BCCheck',
    'TotalRuns', 'TotalWickets', 'BOWLING_LINE_ID', 'BOWLING_LENGTH_ID', 'FiveHaul', 'Flag', 'FlagSet', 'PenaltyRuns',
    'IsFifty', 'IsHundred', 'IsTwoHundred', 'IsHattrick', 'release_speed', 'initial_angle', 'release_x', 'release_y',
    'release_z', 'pre_bounce_ax', 'pre_bounce_ay', 'pre_bounce_az', 'pre_bounce_vx', 'pre_bounce_vy', 'pre_bounce_vz',
    'bounce_angle', 'cof', 'cor', 'pbr', 'shot_attacked', 'shot_played', 'shot_info', 'crease_reaction_time',
    'interception_reaction_time', 'bounce_x', 'bounce_y', 'post_bounce_ax', 'post_bounce_ay', 'post_bounce_az',
    'post_bounce_vx', 'post_bounce_vy', 'post_bounce_vz', 'impact_x', 'impact_y', 'impact_z', 'crease_x', 'crease_y',
    'crease_z', 'drop_angle', 'stump_x', 'stump_y', 'stump_z', 'swing', 'deviation', 'swing_dist', 'six_dist', 'ground',
    'date', 'season'
]

KEY_COLS = ['InningsNo', 'OverNo', 'BallCount']
HAWKEYE_COLS = None  # resolved after first fetch


def load_match_hawk(mapping_path, min_match_id=0):
    match_hawk = {}
    with open(mapping_path, 'r') as f:
        next(f)
        for line in f:
            pair = line.replace(' ', '').strip().split(',')
            if int(pair[0]) >= min_match_id:
                match_hawk[int(pair[0])] = int(pair[1])
    return match_hawk


def process_row(i, match_hawk):
    innings, over, ball, match_id = int(i[0]), int(i[1]), int(i[2]), int(i[3])

    if match_id not in match_hawk:
        print(f"Skipping MatchID {match_id}: not found in mapping")
        return None

    data = fetch_bbb_data(innings, over, ball, match_hawk[match_id])
    if data is None:
        return None

    ball_data = {key: np.nan for key in FIELDS}
    ball_data['MatchID'] = match_id
    ball_data['InningsNo'] = innings
    ball_data['OverNo'] = over
    ball_data['BallCount'] = ball

    processData(ball_data, data)

    if not math.isnan(ball_data['release_speed']):
        return ball_data
    return None


def update_folder(folder, mapping_path, min_match_id=0, max_match_id=None, last_n_matches=None):
    folder = Path(folder)
    print(f"\n--- Processing {folder} ---")

    df = pd.concat(
        [pd.read_csv(file) for file in folder.glob('*.csv')],
        ignore_index=True
    )
    mapped_ids = load_match_hawk(mapping_path, min_match_id).keys()
    df_missing = df[df['MatchID'].isin(mapped_ids) & (df['release_speed'].isna())]
    if max_match_id is not None:
        df_missing = df_missing[df_missing['MatchID'] < max_match_id]

    if last_n_matches is not None:
        top_matches = sorted(df_missing['MatchID'].unique())[-last_n_matches:]
        df_missing = df_missing[df_missing['MatchID'].isin(top_matches)]

    df_short = df_missing[['InningsNo', 'OverNo', 'BallCount', 'MatchID']]
    print(f"Found {len(df_short)} rows missing hawkeye data")

    if df_short.empty:
        return

    if not check_connectivity():
        print("ERROR: Cannot reach polls.iplt20.com — check your network and try again.")
        return

    match_hawk = load_match_hawk(mapping_path, min_match_id)

    # Show missing count per match before fetching
    missing_per_match = df_short.groupby('MatchID').size()
    print(f"{'MatchID':<12} {'Missing rows'}")
    print("-" * 26)
    for mid, count in missing_per_match.items():
        print(f"{mid:<12} {count}")
    print()

    total_data = []
    match_ids = df_short['MatchID'].unique()
    outer_bar = tqdm(match_ids, desc="Matches", position=0)

    for match_id in outer_bar:
        outer_bar.set_description(f"Match {match_id}")
        rows = df_short[df_short['MatchID'] == match_id].values
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_row, i, match_hawk) for i in rows]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  Rows", position=1, leave=False):
                result = future.result()
                if result is not None:
                    total_data.append(result)

    print(f"\nSuccessfully fetched {len(total_data)} rows")

    if not total_data:
        return

    new_df = pd.DataFrame(total_data)
    hawkeye_cols = [c for c in FIELDS if c in new_df.columns and c not in KEY_COLS + ['MatchID']]

    for match_id, group in new_df.groupby('MatchID'):
        path = folder / f'{match_id}.csv'
        orig = pd.read_csv(path)

        merged = orig.merge(
            group[KEY_COLS + hawkeye_cols],
            on=KEY_COLS,
            how='left',
            suffixes=('', '_new'),
        )
        for col in hawkeye_cols:
            new_col = f'{col}_new'
            if new_col in merged.columns:
                merged[col] = merged[new_col].combine_first(merged[col])
                merged.drop(columns=new_col, inplace=True)

        merged.to_csv(path, index=False)
        print(f"Updated {path} with {len(group)} rows")


update_folder(
    folder='bcci_hawkeye_data',
    mapping_path='./bcci_shot_data/Men/hawkeyeid_matchid.csv',
    min_match_id=1896,
)

update_folder(
    folder='bcci_hawkeye_data',
    mapping_path='./bcci_shot_data/Women/hawkeyeid_matchid.csv',
    min_match_id=1235,
)

ipl_last = max(int(f.stem) for f in Path('ipl_hawkeye_data').glob('*.csv'))
update_folder(
    folder='ipl_hawkeye_data',
    mapping_path='./ipl_shot_data/hawkeyeid_matchid.csv',
    min_match_id=2417,   # IPL 2026 start
    max_match_id=ipl_last,  # exclude last (potentially live) match
)