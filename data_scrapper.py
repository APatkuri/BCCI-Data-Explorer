import requests
import pandas as pd
import numpy as np
import re
import base64
import json
import time
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


def fetch_x_api_key():
    url = 'https://polls.iplt20.com/bundle.js?v=1.4'
    headers = {
        'sec-ch-ua-platform': '"Linux"',
        'Referer': 'https://polls.iplt20.com/?entity_matchId=87747&matchId=13390407062092&ipl=1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=100)
        response.raise_for_status()
        # Extract token from the response using regex
        match = re.search(r'"access_token\\":\\"([^"]+)\\"', response.text)
        if match:
            return match.group(1)
        else:
            print("x-api-key not found in response.")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    

def fetch_x_token_key(x_api_key):
    url = f'https://polls.iplt20.com/widget/welcome/get_data?path=matches/87747/innings/info&token=66'
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en;q=0.9',
        'priority': 'u=0, i',
        'referer': 'https://polls.iplt20.com/?entity_matchId=87747&matchId=13390407062092&ipl=1',
        'sec-ch-ua': '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'x-api-key': x_api_key,
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=100)
        response.raise_for_status()
        # Extract the token key from the response
        match = re.search(r'"key":"([^"]+)"', response.text)
        if match:
            return match.group(1)
        else:
            print("x-token-key not found in response.")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    

def fetch_payload(x_api_key, x_token_key, req_url):
    # url = 'https://polls.iplt20.com/widget/welcome/get_data?path=Delivery_1_2_4_13390407062092.json'
    url = req_url
    headers = {
        'accept': '*/*',
        'accept-language': 'en-GB,en;q=0.8',
        'priority': 'u=0, i',
        'referer': 'https://polls.iplt20.com/?entity_matchId=87748&matchId=13390493628543&ipl=1',
        'sec-ch-ua': '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
        'x-api-key': x_api_key,
        'x-token-key': x_token_key,
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=100)
        response.raise_for_status()
        match = re.search(r'"payload":"([^"]+)"', response.text)
        if match:
            return match.group(1)
        else:
            print("Payload not found in response.")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
def decode_payload(payload, key):
    # Decode from base64
    encrypted = base64.b64decode(payload)
    # XOR decryption
    decrypted = ''.join(
        chr(b ^ ord(key[i % len(key)])) for i, b in enumerate(encrypted)
    )
    # Parse JSON
    return json.loads(decrypted)

def fetch_bbb_data(inning, over, ball, hawkID, retry_count=0):

    url = f"{BASE_URL}?path=Delivery_{inning}_{over}_{ball}_{hawkID}.json"
    
    x_api_key = fetch_x_api_key()
    if not x_api_key:
        print("Failed to retrieve x-api-key.")
        return None

    x_token_key = fetch_x_token_key(x_api_key)
    if not x_token_key:
        print("Failed to retrieve x-token-key.")
        return None
    
    payload = fetch_payload(x_api_key, x_token_key, url)

    if payload is None and retry_count < 2:
        print(f"Attempt {retry_count + 1} failed, retrying with fresh tokens...")
        time.sleep(2)
        return fetch_bbb_data(inning, over, ball, hawkID, retry_count + 1)
    
    if not payload:
        print("Failed to retrieve payload.")
        return None
        
    # Decode the payload
    key = "ran_js_my_tok"
    try:
        decoded = decode_payload(payload, key)

        return decoded
        # print("SUCCESS Decoded Data:")
        # print(json.dumps(decoded, indent=2))
    except Exception as e:
        print("FAILED to decode:", str(e))
        return None
        
        

# def fetch_bbb_data(inning, over, ball, hawkID):
#     url = f"{BASE_URL}?path=Delivery_{inning}_{over}_{ball}_{hawkID}.json"

#     try:
#         response = requests.get(url, timeout = 100)
#         data = response.json()
#     except requests.RequestException as e:
#         print(f"Request failed: {e}")
#         return None

#     if not data:
#         return None
    
#     return data

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

        

    


