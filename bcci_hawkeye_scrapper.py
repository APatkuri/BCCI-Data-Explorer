# import csv
from data_scrapper import *
from process_data import *
import numpy as np
import pandas as pd
import streamlit as st
import os

FIELDS = ['MatchID', 'InningsNo', 'BattingTeamID',
          'TeamName', 'BatsManName', 'BowlerName', 'BowlerType', 'OverNo',
          'BallNo', 'Runs', 'BallRuns', 'ActualRuns', 'IsOne', 'IsTwo', 'IsThree',
          'IsDotball', 'Extras', 'IsWide', 'IsNoBall', 'IsBye', 'IsLegBye',
          'IsFour', 'IsSix', 'IsWicket', 'WicketType', 'Wickets',
          'IsBowlerWicket', 'BallName', 'Day', 'SESSION_NO', 'IsExtra', 'SNO',
          'Xpitch', 'Ypitch', 'RunRuns', 'IsMaiden', 'OverImage', 'BowlTypeID',
          'BowlTypeName', 'ShotTypeID', 'ShotType', 'IsBouncer', 'IsFreeHit',
          'BallCount', 'BCCheck', 'TotalRuns', 'TotalWickets', 'BOWLING_LINE_ID',
          'BOWLING_LENGTH_ID', 'FiveHaul', 'Flag', 'FlagSet', 'PenaltyRuns',
          'IsFifty', 'IsHundred', 'IsTwoHundred', 'IsHattrick',

          'release_speed', 'initial_angle', 'release_x', 'release_y', 'release_z', 'pre_bounce_ax', 'pre_bounce_ay', 'pre_bounce_az', 
          'pre_bounce_vx', 'pre_bounce_vy', 'pre_bounce_vz', 'bounce_angle', 
          
          'cof', 'cor', 'pbr', 'shot_attacked', 'shot_played', 'shot_info', 'crease_reaction_time', 'interception_reaction_time',

          'bounce_x', 'bounce_y', 'post_bounce_ax', 'post_bounce_ay',
          'post_bounce_az', 'post_bounce_vx', 'post_bounce_vy', 'post_bounce_vz', 'impact_x', 'impact_y', 'impact_z', 'crease_x', 
          'crease_y', 'crease_z', 'drop_angle', 'stump_x', 'stump_y', 'stump_z', 'swing', 'deviation', 'swing_dist', 'six_dist', 
          'ground', 'date', 'season']

def hawkeye_main(cat, matchid, hawkeyeid):
    placeholder = st.empty()
    # Reading t20bbb for validation since we don't know the max number of balls (extras + legal balls) in an over 
    # MATCH_INFO_FILENAME = 'matches/hawkeye-keydatatest.csv'
    # hawkeye_ids, match_ids, seasons = read_match_ids(MATCH_INFO_FILENAME)
    hawkeye_ids = [hawkeyeid]
    match_ids = [matchid]
    all_match_info_df = pd.read_csv(f"./bcci_shot_data/{cat}/bcci_match_list.csv")

    match_info_df = all_match_info_df[all_match_info_df['MatchID'] == matchid]
    max_over = match_info_df['MATCH_NO_OF_OVERS'].max().astype(int)
    seasons = ['']

    max_innings = 4 if match_info_df['MATCH_NO_OF_OVERS'].max() == 200 else 2


    # iterate over the matches
    for i in range(len(hawkeye_ids)):
        OUT_FILENAME = f"./bcci_hawkeye_data/{match_ids[i]}.csv"
        ball_data_all = []

        hawkID = hawkeye_ids[i]
        matchID = match_ids[i]
        season = seasons[i]
        
        # match_str = "t20s_csv2/" + str(matchID) + ".csv"
        # match_str = "t20_" + str(season) + ".csv"
        match_str = f"./bcci_shot_data/{cat}/csv/{matchID}.csv"

        try:
            t20bbb = pd.read_csv(match_str, dtype={'line': str, 'length': str, 'shot': str})
        except FileNotFoundError:
            print(f"Error: File '{match_str}' not found. Skipping this match.")
            continue

        # max_overs_per_innings = t20bbb.groupby('InningsNo')['OverNo'].max()
        max_innings = t20bbb['InningsNo'].max()
        max_overs_per_innings = t20bbb.groupby('InningsNo')['OverNo'].max()

        # last_rows_per_innings = t20bbb.groupby('InningsNo').tail(1)
        # last_rows_per_innings = last_rows_per_innings[['InningsNo', 'OverNo', 'BallNo']]

        # t20bbb = pd.read_csv(match_str, dtype = {'line': str, 'length': str, 'shot': str})
        if os.path.exists(OUT_FILENAME):
            # If it exists, read the existing data into a DataFrame
            hawkeye_data = pd.read_csv(OUT_FILENAME)
            if len(hawkeye_data) == len(t20bbb):
                print("Data already processed, stopping function.")
                return

            last_row = hawkeye_data.iloc[-1]
            start_inning_no = last_row['InningsNo']
            start_over_no = last_row['OverNo']
            start_ball_no = last_row['BallCount']
            count = len(hawkeye_data)
        else:
            # If it doesn't exist, initialize an empty list for new data
            hawkeye_data = pd.DataFrame(columns=FIELDS)
            start_inning_no = 1
            start_over_no = 1
            start_ball_no = 0

            count = 0

        count = 0

        for inning in range(start_inning_no, max_innings+1):
            # if inning not in max_overs_per_innings:
            #     continue
            max_over = max_overs_per_innings.get(inning, 0)
            if(inning == start_inning_no):
               start_over_no = start_over_no
            else:
                start_over_no = 1

            for over in range(start_over_no, max_over+1):

                if((inning == start_inning_no) & (over == start_over_no)):
                    ball = start_ball_no+1
                else:
                    ball = 1

                while True:
                    # initialize ball_data dictionary to add values for keys for FIELDS above
                    ball_data = {key: np.nan for key in FIELDS}
                    ball_data['season'] = seasons[i]

                    # fetch data from api and data from 2nd dataset for checking attributes
                    data = fetch_bbb_data(inning, over, ball, hawkID)
                    ball_id = int(over) - 1 + float(ball) / 10
                    # ball_id = int(over) - 1 + float(ball) / 100

                    print(inning, ball_id)
                    placeholder.text(f"Inning No: {inning}, Ball: {ball_id}")
                    # ball_data_checks = t20bbb.loc[(t20bbb['match_id'] == int(matchID)) & (t20bbb['ball'] == ball_id) & (t20bbb['innings'] == inning)]
                    # ball_data_checks = t20bbb.loc[(t20bbb['p_match'] == int(matchID)) & (t20bbb['ball_id'] == ball_id) & (t20bbb['inns'] == inning)]
                    ball_data_checks = t20bbb.loc[(t20bbb['MatchID'] == int(matchID)) & (t20bbb['OverNo'] == over) & (t20bbb['BallCount'] == ball) & (t20bbb['InningsNo'] == inning)]
                    # ball_data_checks = pd.DataFrame()

                    # no data fetched
                    if not data:
                        # either no data OR api is not available for that delivery (imputation)
                        if ball_data_checks.empty:
                            break
                        else:
                            # Fill attributes not associated with hawkeye
                            fill_non_hawkeye_data(ball_data, ball_data_checks)
                            print(f'IMPUTATION DONE - {matchID} {inning} {over} {ball}')
                            placeholder.text(f'IMPUTATION DONE - MatchID: {matchID}, InningNo: {inning}, OverNo: {over}, BallNo: {ball}')
                    else:
                    #     # Fill non hawkeye and hawkeye attributes
                        if ball_data_checks.empty:
                            break
                            # processData(ball_data, data)
                        else:
                            fill_non_hawkeye_data(ball_data, ball_data_checks)
                            processData(ball_data, data)
                        # print(ball_data)

                    ball_data_all.append(ball_data)

                    ball += 1
                    count += 1

        # new_data_df = pd.DataFrame(ball_data_all)
        new_hawkeye_data = pd.DataFrame(ball_data_all)
        # hawkeye_data = pd.concat([hawkeye_data, new_hawkeye_data], ignore_index=True)
        new_hawkeye_data.to_csv(OUT_FILENAME, mode='a', header=not os.path.exists(OUT_FILENAME), index=False)
        # hawkeye_data.to_csv(OUT_FILENAME, index = False)       
        print(f'{matchID} {count} done')
        placeholder.empty()



# if __name__ == "__main__":
#     hawkeye_main()
