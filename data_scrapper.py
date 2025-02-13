import requests
import pandas as pd
import numpy as np
# from bcci_hawkeye_scrapper import FIELDS

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

BASE_URL = "https://polls.iplt20.com/widget/welcome/get_data"

def fetch_bbb_data(inning, over, ball, hawkID):
    url = f"{BASE_URL}?path=Delivery_{inning}_{over}_{ball}_{hawkID}.json"

    try:
        response = requests.get(url, timeout = 100)
        data = response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if not data:
        return None
    
    return data

def fill_non_hawkeye_data(ball_data: dict, ball_data_check: pd.DataFrame):
    '''
        Fills non hawkeye data attributes for the ball
    '''
    
    # non_hawkeye_attr = FIELDS[ : FIELDS.index('other_player_dismissed') + 1]
    # non_hawkeye_attr = FIELDS[ : FIELDS.index('control') + 1]
    non_hawkeye_attr = FIELDS[ : FIELDS.index('IsHattrick') + 1]
    
    for attribute in non_hawkeye_attr:
        
        if attribute == "bowl_type":
            ball_data[attribute] = ball_data_check["bowl_kind"].values[0].split()[0]
        elif attribute == "ground":
            ground = ball_data_check.iloc[0]['ground']
            ball_data[attribute] = "-".join(ground.replace(",", " ").split()).upper()
        elif attribute == "team_bat":
            team_bat = ball_data_check.iloc[0]['team_bat']
            ball_data[attribute] = "-".join(team_bat.split())
        elif attribute == "team_bowl":
            team_bowl = ball_data_check.iloc[0]['team_bowl']
            ball_data[attribute] = "-".join(team_bowl.split())
        else:
            # ball_data[attribute] = ball_data_check[attribute].values[0]
            ball_data[attribute] = ball_data_check.get(attribute, pd.Series([""])).values[0]

        

    


