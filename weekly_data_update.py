from bcci_shot_data import main_func
from bcci_hawkeye_scrapper import hawkeye_main
import os
import pandas as pd

def get_hawkeye_data():
    
    men_df = pd.read_csv('./bcci_shot_data/Men/hawkeyeid_matchid.csv', low_memory=False)
    women_df = pd.read_csv('./bcci_shot_data/Women/hawkeyeid_matchid.csv', low_memory=False)
    match_list = men_df['MatchID'].to_list() + women_df['MatchID'].to_list()

    curr_match_list = os.listdir('./bcci_hawkeye_data')
    curr_match_list = [int(x[:-4]) for x in curr_match_list if x.endswith('.csv')]

    res = [x for x in match_list if x not in curr_match_list]

    for gender in ["Men", "Women"]:
        live_file = f"./bcci_shot_data/{gender}/live_data_file_name.txt"
        if os.path.exists(live_file):
            with open(live_file, "r") as f:
                live_matches = [int(line.strip()) for line in f if line.strip().isdigit()]
            res = list(set(res + live_matches))

    if not res:
        return
    
    for match_id in res:
        if match_id in men_df['MatchID'].values:
            hawkeye_id = men_df.loc[men_df['MatchID'] == match_id, 'HawkeyeID'].values[0]
            hawkeye_main('Men', match_id, hawkeye_id, 'bcci')
        elif match_id in women_df['MatchID'].values:
            hawkeye_id = women_df.loc[women_df['MatchID'] == match_id, 'HawkeyeID'].values[0]
            hawkeye_main('Women', match_id, hawkeye_id, 'bcci')

if __name__ ==  "__main__":
    main_func("Men")
    main_func("Women")
    get_hawkeye_data()