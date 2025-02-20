import streamlit as st
import os
import plotly.graph_objects as go
import sys

parent_dir_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir_1 not in sys.path:
    sys.path.append(parent_dir_1)

from bcci_shot_data import main_func
from bcci_hawkeye_scrapper import hawkeye_main
from pitch_view.pitch_densitymap import *

import pandas as pd
import matplotlib.pyplot as plt

st.title("BCCI Batting Playground")

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

submit_button = st.button('Update Shot Data')

cat = st.selectbox('Category', ['Men', 'Women'])

if submit_button:
    with st.spinner("Updating Shot data... Please wait."):
        main_func(cat)
    st.success("Update completed successfully!")

try:
    shot_data_df = pd.read_csv(f"./bcci_shot_data/{cat}/combined_shot_data.csv", low_memory=False)
    match_list_df = pd.read_csv(f"./bcci_shot_data/{cat}/bcci_match_list.csv", low_memory=False)

    match_list_df = match_list_df[match_list_df['HomeTeamID'].isin([1,52])]
except:
    st.error("Cannot Load Data")

format_type = st.selectbox(
    'Format',
    ['Test', 'ODI', 'T20'],
    index=None,
    placeholder='Choose an option'
)

format_df = match_list_df[match_list_df['MatchTypeName'] == format_type]

unique_competitions = format_df[['CompetitionID', 'CompetitionName']].drop_duplicates()
unique_competitions['Display'] = unique_competitions.apply(lambda x: f"{x['CompetitionName']} ({x['CompetitionID']})", axis=1)

series_name = st.selectbox(
    'Series',
    # format_df['CompetitionName'].unique(),
    unique_competitions['Display'],
    index=None,
    placeholder='Choose an option',
)

if series_name:
    selected_competition_id = int(series_name.split("(")[-1][:-1])
    match_name = st.selectbox(
        'Match',
        # format_df[format_df['CompetitionName'] == series_name]['MatchOrder'],
        format_df[format_df['CompetitionID'] == selected_competition_id]['MatchOrder'],
        index=None,
        placeholder='Choose an option',
    )

def shot_type_dist(match_shot_data):

    fig = go.Figure()

    for bowler in match_shot_data['BowlerName'].unique():
        bowler_data = match_shot_data[(match_shot_data['BowlerName'] == bowler)]
        
        # Create the histogram trace for each bowler
        fig.add_trace(go.Histogram(
            x=[y for y in bowler_data['ShotType'] if y!=None],
            histnorm='probability',
            name=f'{bowler}', # name used in legend and hover labels
            opacity=0.75
        ))

    fig.update_layout(
        title_text=f'Shot Type Probability Distribution', # title of plot
        xaxis_title_text=f'{type}', # xaxis label
        yaxis_title_text='Probability', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
        # xaxis=dict(
        #     tickmode='array',
        #     tickvals=custom_order,  # Set the categories
        #     ticktext=custom_order,  # Set the text labels for those categories
        #     categoryorder='array',  # Specify that the order will be an array
        #     categoryarray=custom_order  # Provide the custom order of categories
        # )        
    )

    st.plotly_chart(fig)

def batting_hawkeye_plots(custom_df, plotno):

    max_over_limit = custom_df['OverNo'].max()
    min_over_limit = custom_df['OverNo'].min()

    name_list = list(custom_df['BowlerName'].unique())
    # batter_name = custom_df['']

    def show_stump(ax):
        stump_height = 0.711
        stump_width = 0.2286 / 2
        stump_positions = [-stump_width, 0, stump_width]
        for pos in stump_positions:
            ax.plot([pos, pos], [0, stump_height], color='brown', linewidth=3)
        bail_y = stump_height  # Bails are at the top of the stumps
        ax.plot([stump_positions[0], stump_positions[1]], [bail_y, bail_y], color='brown', linewidth=2)  # Off-stump to Middle-stump
        ax.plot([stump_positions[1], stump_positions[2]], [bail_y, bail_y], color='brown', linewidth=2)  # Middle-stump to Leg-stump

    if(plotno == 2):
        for name in name_list:
            df_bowler = custom_df[custom_df['BowlerName'].str.contains(f"{name}", case=False, na=False)]
            bowler_name = df_bowler['BowlerName'].unique()[0]

            if(len(df_bowler) > 0):
                release_y_list = [float("%.2f"%y) for y, z in zip(df_bowler['stump_y'], df_bowler['stump_z']) if -50<y<50 and -50<z<50]
                release_z_list = [float("%.2f"%z) for y, z in zip(df_bowler['stump_y'], df_bowler['stump_z']) if -50<y<50 and -50<z<50]
                balls_hitting = [(y, z) for y,z in zip(release_y_list, release_z_list) if (z<0.711)&(y<0.1143)&(y>(-0.1143))]
                perc_hitting = (len(balls_hitting)*100)/len(release_y_list)
                show_stump(plt)
                plt.plot(release_y_list,release_z_list, 'o',label=f"{bowler_name} {perc_hitting:.2f}%")
                plt.xlim(-1.75, 1.75)
                plt.ylim(0, 2.5)
                plt.title(f"{batter_name}: Overs {min_over_limit}-{max_over_limit} Beehive Bowler Wise")
                plt.legend()
                plt.grid(True, linestyle='--')
    
    elif(plotno == 3):
        custom_df['shot_played'] = custom_df['shot_played'].fillna("Played")
        custom_df['shot_attacked'] = custom_df['shot_attacked'].fillna(custom_df['ShotType'].apply(lambda x: 'Defended' if any(word in str(x) for word in ['Alone', 'Defended', 'Defence']) else 'Attacked'))
        # unique_pairs = custom_df[['shot_attacked', 'shot_played']].drop_duplicates().values
        shot_attack_types = list(custom_df['shot_attacked'].unique())
        shot_attack_types.append("Missed/Edged")
        total_shots = len(custom_df)

        for shot_attack in shot_attack_types:

            if shot_attack == "Missed/Edged":
                df_shot_attacked = custom_df[(custom_df['shot_played'].isin(['Edged', 'Missed']))]
            else:
                df_shot_attacked = custom_df[(custom_df['shot_attacked'].str.contains(f"{shot_attack}", case=False, na=False)) & (custom_df['shot_played'].str.contains("Played", case=False, na=False))]
            # shot_attack_type = df_shot_attacked['shot_attacked'].unique()[0]
            if(len(df_shot_attacked)):
                release_y_list = [float("%.2f"%y) for y, z in zip(df_shot_attacked['stump_y'], df_shot_attacked['stump_z']) if -50<y<50 and -50<z<50]
                release_z_list = [float("%.2f"%z) for y, z in zip(df_shot_attacked['stump_y'], df_shot_attacked['stump_z']) if -50<y<50 and -50<z<50]
                show_stump(plt)
                shot_attacked_number = len(df_shot_attacked)
                # print(shot_attacked_number)
                plt.plot(release_y_list,release_z_list, 'o',label=f"{shot_attack} {shot_attacked_number}/{total_shots}")  

                # if(shot_play == "Played"):
                    # plt.plot(release_y_list,release_z_list, 'o', alpha=0.6,label=f"{shot_attack} {shot_attacked_number}/{total_shots}")    
                # else:
                    # plt.plot(release_y_list,release_z_list, 'o',label=f"{shot_attack} {shot_play} {shot_attacked_number}/{total_shots}")
                plt.xlim(-1.75, 1.75)
                plt.ylim(0, 2.5)
                plt.title(f"{batter_name}: Overs {min_over_limit}-{max_over_limit} Beehive Control Wise")
                plt.legend()
                plt.grid(True, linestyle='--')
        
    elif(plotno == 4):
        beehive_df = custom_df[custom_df['stump_y'].between(-50, 50) & custom_df['stump_z'].between(-50, 50)].copy()
        boundaries_df = beehive_df[((beehive_df['IsFour'] == 1) | (beehive_df['IsSix'] == 1)) & (beehive_df['IsWicket'] == 0)]
        wickets_df = beehive_df[(beehive_df['IsWicket'] == 1)]
        dots_df = beehive_df[(beehive_df['IsDotball'] == 1) & (beehive_df['IsWicket'] == 0)]
        runs_df = beehive_df[pd.to_numeric(beehive_df['BallRuns'], errors='coerce').fillna(0).astype(int).gt(0) & (beehive_df['IsFour'] == 0) & (beehive_df['IsSix'] == 0) & (beehive_df['IsWicket'] == 0)]
        show_stump(plt)
        plt.plot(dots_df['stump_y'],dots_df['stump_z'], 'o', color='green',label="Dots", alpha=0.5)
        plt.plot(runs_df['stump_y'],runs_df['stump_z'], 'o', color='yellow',label="Runs", alpha=0.5)
        plt.plot(boundaries_df['stump_y'],boundaries_df['stump_z'], 'o', color='red',label="4s/6s")
        plt.plot(wickets_df['stump_y'],wickets_df['stump_z'], 'o', color='blue',label="Wicket")
        plt.title(f"{batter_name}: Overs {min_over_limit}-{max_over_limit} Beehive Outcome Wise")
        plt.xlim(-1.75, 1.75)
        plt.ylim(0, 2.5)
        plt.legend()
        plt.grid(True, linestyle='--')

    elif(plotno == 5):
        title = f'{batter_name}'
        subtitle_1 = f'{series_name}: {match_name}'
        subtitle_2 = f'Tracking enabled for {len(custom_df)} balls between Overs {min_over_limit}-{max_over_limit}.'
        fig = plot_pitch(custom_df, title, subtitle_1, subtitle_2)

    elif(plotno == 6):
        title = f'{batter_name}'
        subtitle_1 = f'{series_name}: {match_name}'
        subtitle_2 = f'Tracking enabled for {len(custom_df)} balls between Overs {min_over_limit}-{max_over_limit}.'
        fig = plot_control_pitch(custom_df, title, subtitle_1, subtitle_2)

    elif(plotno == 7):
        beehive_df = custom_df[custom_df['bounce_x'] >= 0].copy()
        boundaries_df = beehive_df[((beehive_df['IsFour'] == 1) | (beehive_df['IsSix'] == 1)) & (beehive_df['IsWicket'] == 0)]
        wickets_df = beehive_df[(beehive_df['IsWicket'] == 1)]
        dots_df = beehive_df[(beehive_df['IsDotball'] == 1) & (beehive_df['IsWicket'] == 0)]
        runs_df = beehive_df[pd.to_numeric(beehive_df['BallRuns'], errors='coerce').fillna(0).astype(int).gt(0) & (beehive_df['IsFour'] == 0) & (beehive_df['IsSix'] == 0) & (beehive_df['IsWicket'] == 0)]
        title = f'{batter_name}'
        subtitle_1 = f'{series_name}: {match_name}'
        subtitle_2 = f'Tracking enabled for {len(custom_df)} balls between Overs {min_over_limit}-{max_over_limit}.'
        fig = pitch_map(dots_df, runs_df, boundaries_df, wickets_df, title, subtitle_1, subtitle_2)
    
    elif(plotno == 8):
        for name in name_list:
            df_bowler = custom_df[custom_df['BowlerName'].str.contains(f"{name}", case=False, na=False)]
            bowler_name = df_bowler['BowlerName'].unique()[0]

            if(len(df_bowler) > 0):
                release_y_list = [-float("%.2f"%(stump_x - x)) for x, stump_x, z in zip(df_bowler['impact_x'], df_bowler['stump_x'] ,df_bowler['impact_z']) if -50<x<50 and 0<z<50 and -50<stump_x<50]
                release_z_list = [float("%.2f"%z) for x, stump_x, z in zip(df_bowler['impact_x'], df_bowler['stump_x'],df_bowler['impact_z']) if -50<x<50 and 0<z<50 and -50<stump_x<50]
                # balls_hitting = [(y, z) for y,z in zip(release_y_list, release_z_list) if (z<0.711)&(y<0.1143)&(y>(-0.1143))]
                # perc_hitting = (len(balls_hitting)*100)/len(release_y_list)
                plt.plot([0, 0], [0, 0.711], color='brown', linewidth=10)
                # show_stump(plt)
                plt.plot(release_y_list,release_z_list, 'o',label=f"{bowler_name}")
                plt.xlim(0, 5)
                plt.ylim(0, 2)
                plt.title(f"{batter_name}: Overs {min_over_limit}-{max_over_limit} Impact Points Bowler Wise")
                plt.legend()
                plt.grid(True, linestyle='--')
    
    elif(plotno == 9):
        custom_df['shot_played'] = custom_df['shot_played'].fillna("Played")
        custom_df['shot_attacked'] = custom_df['shot_attacked'].fillna(custom_df['ShotType'].apply(lambda x: 'Defended' if any(word in str(x) for word in ['Alone', 'Defended', 'Defence']) else 'Attacked'))
        # unique_pairs = custom_df[['shot_attacked', 'shot_played']].drop_duplicates().values
        shot_attack_types = list(custom_df['shot_attacked'].unique())
        shot_attack_types.append("Missed/Edged")
        total_shots = len(custom_df)

        for shot_attack in shot_attack_types:

            if shot_attack == "Missed/Edged":
                df_shot_attacked = custom_df[(custom_df['shot_played'].isin(['Edged', 'Missed']))]
            else:
                df_shot_attacked = custom_df[(custom_df['shot_attacked'].str.contains(f"{shot_attack}", case=False, na=False)) & (custom_df['shot_played'].str.contains("Played", case=False, na=False))]
            # shot_attack_type = df_shot_attacked['shot_attacked'].unique()[0]
            if(len(df_shot_attacked)):
                release_y_list = [-float("%.2f"%(stump_x - x)) for x, stump_x, z in zip(df_shot_attacked['impact_x'], df_shot_attacked['stump_x'] ,df_shot_attacked['impact_z']) if -50<x<50 and 0<z<50 and -50<stump_x<50]
                release_z_list = [float("%.2f"%z) for x, stump_x, z in zip(df_shot_attacked['impact_x'], df_shot_attacked['stump_x'],df_shot_attacked['impact_z']) if -50<x<50 and 0<z<50 and -50<stump_x<50]
                plt.plot([0, 0], [0, 0.711], color='brown', linewidth=10)
                shot_attacked_number = len(df_shot_attacked)
                # print(shot_attacked_number)
                plt.plot(release_y_list,release_z_list, 'o',label=f"{shot_attack} {shot_attacked_number}/{total_shots}")  

                # if(shot_play == "Played"):
                    # plt.plot(release_y_list,release_z_list, 'o', alpha=0.6,label=f"{shot_attack} {shot_attacked_number}/{total_shots}")    
                # else:
                    # plt.plot(release_y_list,release_z_list, 'o',label=f"{shot_attack} {shot_play} {shot_attacked_number}/{total_shots}")
                plt.xlim(0, 5)
                plt.ylim(0, 2)
                plt.title(f"{batter_name}: Overs {min_over_limit}-{max_over_limit} Impact Points Control Wise")
                plt.legend()
                plt.grid(True, linestyle='--')

    elif(plotno == 10):
        # custom_df['impact_stump_x_diff'] = custom_df['impact_x'] - custom_df['stump_x']
        custom_df.loc[:, 'impact_stump_x_diff'] = custom_df['impact_x'] - custom_df['stump_x']
        beehive_df = custom_df[custom_df['impact_x'].between(-50, 50) & custom_df['stump_x'].between(-50, 50) & custom_df['impact_z'].between(-50, 50)].copy()
        boundaries_df = beehive_df[((beehive_df['IsFour'] == 1) | (beehive_df['IsSix'] == 1)) & (beehive_df['IsWicket'] == 0)]
        wickets_df = beehive_df[(beehive_df['IsWicket'] == 1)]
        dots_df = beehive_df[(beehive_df['IsDotball'] == 1) & (beehive_df['IsWicket'] == 0)]
        runs_df = beehive_df[pd.to_numeric(beehive_df['BallRuns'], errors='coerce').fillna(0).astype(int).gt(0) & (beehive_df['IsFour'] == 0) & (beehive_df['IsSix'] == 0) & (beehive_df['IsWicket'] == 0)]
        # show_stump(plt)
        if not dots_df.empty:
            plt.plot(dots_df['impact_stump_x_diff'],dots_df['impact_z'], 'o', color='green',label="Dots", alpha=0.5)
        if not runs_df.empty:
            plt.plot(runs_df['impact_stump_x_diff'],runs_df['impact_z'], 'o', color='yellow',label="Runs", alpha=0.5)
        if not boundaries_df.empty:
            plt.plot(boundaries_df['impact_stump_x_diff'],boundaries_df['impact_z'], 'o', color='red',label="4s/6s")
        if not wickets_df.empty:
            plt.plot(wickets_df['impact_stump_x_diff'],wickets_df['impact_z'], 'o', color='blue',label="Wicket")
        plt.plot([0, 0], [0, 0.711], color='brown', linewidth=10)
        plt.xlim(0, 5)
        plt.ylim(0, 2)
        plt.title(f"{batter_name}: Overs {min_over_limit}-{max_over_limit} Impact Points Outcome Wise")
        plt.legend()
        plt.grid(True, linestyle='--')
    
    st.pyplot(plt)


if(format_type and series_name and match_name):

    match_df = format_df[(format_df['CompetitionID'] == selected_competition_id) & (format_df['MatchOrder'] == match_name)]
    match_id = match_df['MatchID'].unique()[0]
    max_overs = match_df['MATCH_NO_OF_OVERS'].unique()[0]

    match_shot_data = shot_data_df[shot_data_df['MatchID'] == match_id]
    batter_list = match_shot_data['BatsManName'].dropna().unique()
    max_len_shot_data = len(match_shot_data)

    batter_name = st.selectbox(
        'Batter',
        batter_list,
        index=None,
        placeholder='Choose a batter'
    )

    if batter_name:

        over_range = st.slider(
            "Overs", 0, max_overs, (0, max_overs)
        )

        match_shot_data = match_shot_data[match_shot_data['BatsManName'] == batter_name]
        innings_list = match_shot_data['InningsNo'].unique()
        innings_type = st.multiselect(
                'Innings',
                innings_list,
                default=innings_list,
                placeholder='Choose an option'
        )
        match_shot_data = match_shot_data[match_shot_data['InningsNo'].isin(innings_type)]

        match_shot_bowler_list = match_shot_data['BowlerName'].dropna().unique()
        bowlers_name = st.multiselect(
            'Bowlers',
            match_shot_bowler_list,
            default=match_shot_bowler_list,
            placeholder='Choose an option'
        )

        if(bowlers_name):

            try:
                hawk_eye_df = pd.read_csv(f"./bcci_hawkeye_data/{match_id}.csv", low_memory=False)
                max_len_hawk_eye_data = len(hawk_eye_df)
                # max_hawkeye_inns = hawk_eye_df['InningsNo'].max()
                # max_hawkeye_overs = hawk_eye_df['OverNo'].max()
                hawk_eye_df = hawk_eye_df[(hawk_eye_df['OverNo'].between(over_range[0], over_range[1])) & (hawk_eye_df['BatsManName'] == batter_name)]
                # hawk_eye_bowler_list = hawk_eye_df['BowlerName'].dropna().unique()

                # bowlers_name = st.multiselect(
                #     'Bowlers',
                #     hawk_eye_bowler_list,
                #     default=hawk_eye_bowler_list,
                #     placeholder='Choose an option'
                # )
                
                hawkeye_batter_df = hawk_eye_df[hawk_eye_df['BowlerName'].isin(bowlers_name) & hawk_eye_df['InningsNo'].isin(innings_type)]
                hawkid_matchid_df = pd.read_csv(f"./bcci_shot_data/{cat}/hawkeyeid_matchid.csv", low_memory=False)
                hawkeye_available = hawkid_matchid_df["MatchID"].isin([match_id])
                # if((max_hawkeye_inns < max_shot_data_inns) or (max_hawkeye_overs < max_shot_data_overs)):
                # print(max_len_shot_data, max_len_hawk_eye_data)
                if(max_len_hawk_eye_data < max_len_shot_data):
                    # st.success("Hawkeye can be Updated!")
                    available_hawkeye_id = hawkid_matchid_df[hawkid_matchid_df['MatchID'] == match_id]["HawkeyeID"].unique()[0]
                    if st.button("Update Hawkeye Data"):
                        with st.spinner("Fetching Hawkeye data... Please wait."):
                            hawkeye_main(cat, match_id, available_hawkeye_id)
                        # Call your function here, e.g., `your_function(hawk_eye_df)`
                        # st.write("Fetching Hawkeye data...")

                        st.success("Hawkeye data fetching completed successfully!")

                # speed_data(hawkeye_bowling_df)
            except:
                hawk_eye_df = None
                hawkid_matchid_df = pd.read_csv(f"./bcci_shot_data/{cat}/hawkeyeid_matchid.csv", low_memory=False)
                hawkeye_available = hawkid_matchid_df["MatchID"].isin([match_id])

                if(hawkeye_available.any()):
                    st.success("Hawkeye is Available")
                    available_hawkeye_id = hawkid_matchid_df[hawkid_matchid_df['MatchID'] == match_id]["HawkeyeID"].unique()[0]
                    if st.button("Get Hawkeye Data"):
                        with st.spinner("Fetching Hawkeye data... Please wait."):
                            hawkeye_main(cat, match_id, available_hawkeye_id)
                        # Call your function here, e.g., `your_function(hawk_eye_df)`
                        # st.write("Fetching Hawkeye data...")
                        st.success("Hawkeye data fetching completed successfully!")

                else:
                    st.warning("Hawkeye Data Not Available")
                
            final_match_shot_data = match_shot_data[(match_shot_data['OverNo'].between(over_range[0], over_range[1])) & (match_shot_data['BowlerName'].isin(bowlers_name))]
            available_shot_data = final_match_shot_data["ShotType"].notna().any()
            if(available_shot_data == False):
                st.warning("Shot Data Not Available")
        else:
            st.warning("Select atlest one bowler")


if (format_type and series_name and match_name and batter_name and bowlers_name and available_shot_data and len(final_match_shot_data)> 0):

    if(hawk_eye_df is not None):
        # print("1")
        selected_option = st.selectbox('Choose an option', ['Shot Type Probability Distribution', 
                                                            'Beehive Bowler Wise', 'Beehive Control Wise', 'Beehive Outcome Wise',  
                                                            'PitchMap Bowler Wise', 'PitchMap Control Wise', 'PitchMap Outcome Wise',
                                                            'Impact Points Bowler Wise', 'Impact Points Control Wise', 'Impact Points Outcome Wise'])
    else:
        selected_option = st.selectbox('Choose an option', ['Shot Type Probability Distribution'])

    def plotting_func(selected_option):
        if(selected_option == 'Shot Type Probability Distribution'):
            shot_type_dist(final_match_shot_data)
        elif(selected_option == 'Beehive Bowler Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 2)
        elif(selected_option == 'Beehive Control Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 3)
        elif(selected_option == 'Beehive Outcome Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 4)
        elif(selected_option == 'PitchMap Bowler Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 5)
        elif(selected_option == 'PitchMap Control Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 6)
        elif(selected_option == 'PitchMap Outcome Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 7)
        elif(selected_option == 'Impact Points Bowler Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 8)
        elif(selected_option == 'Impact Points Control Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 9)
        elif(selected_option == 'Impact Points Outcome Wise'):
            batting_hawkeye_plots(hawkeye_batter_df, 10)

    if(selected_option):
        plotting_func(selected_option)