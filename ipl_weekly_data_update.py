from ipl_shot_data import main_func
from bcci_hawkeye_scrapper import hawkeye_main
import os
import pandas as pd

def get_hawkeye_data():
    
    ipl_df = pd.read_csv('./ipl_shot_data/hawkeyeid_matchid.csv', low_memory=False)
    match_list = ipl_df['MatchID'].to_list()

    curr_match_list = os.listdir('./ipl_hawkeye_data')
    curr_match_list = [int(x[:-4]) for x in curr_match_list if x.endswith('.csv')]

    res = [x for x in match_list if x not in curr_match_list]

    live_file = f"./ipl_shot_data/live_data_file_name.txt"
    if os.path.exists(live_file):
        with open(live_file, "r") as f:
            live_matches = [int(line.strip()) for line in f if line.strip().isdigit()]
        res = list(set(res + live_matches))

    if not res:
        return
    
    for match_id in res:
        if match_id in ipl_df['MatchID'].values:
            hawkeye_id = ipl_df.loc[ipl_df['MatchID'] == match_id, 'HawkeyeID'].values[0]
            hawkeye_main('Test', match_id, hawkeye_id, 'ipl')

if __name__ ==  "__main__":
    get_hawkeye_data()
    main_func()
    get_hawkeye_data()