import streamlit as st
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
from bcci_shot_data import main_func
from bcci_hawkeye_scrapper import hawkeye_main
from pitch_view.pitch_densitymap import *

st.title("BCCI Data Playground")

##################
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

submit_button = st.button('Update Shot Data')

cat = st.selectbox('Category', ['Men', 'Women'])

if submit_button:
    with st.spinner("Updating Shot data... Please wait."):
        main_func(cat)
    st.success("Update completed successfully!")
    

# submit_button = st.button("Update Data")
# if submit_button:
#     run_task()

############


#############
try:
    shot_data_df = pd.read_csv(f"./bcci_shot_data/{cat}/combined_shot_data.csv", low_memory=False)
    match_list_df = pd.read_csv(f"./bcci_shot_data/{cat}/bcci_match_list.csv", low_memory=False)
except:
    st.error("Cannot Load Data")

format_type = st.selectbox(
    'Format',
    ['Test', 'ODI', 'T20'],
    index=None,
    placeholder='Choose an option'
)

format_df = match_list_df[match_list_df['MatchTypeName'] == format_type]


series_name = st.selectbox(
    'Series',
    format_df['CompetitionName'].unique(),
    index=None,
    placeholder='Choose an option',
)

match_name = st.selectbox(
    'Match',
    format_df[format_df['CompetitionName'] == series_name]['MatchOrder'],
    index=None,
    placeholder='Choose an option',
)

# if(format_type):
#     temp = format_df['CompetitionName'] + " " + format_df['MatchOrder']
#     select_match = st.selectbox(
#         'Match',
#         temp,
#         index=None,
#         placeholder='Choose an option'
#     )

##############################

def line_length_dist(match_shot_data, type):

    if type == "Length":
        req_str = 'BOWLING_LENGTH_ID'
        custom_order = ['Full Toss', 'Yorker', 'Full Length', 'Good Length', 'Short of Good Length', 'Short Length']
    elif type == "Line":
        req_str = 'BOWLING_LINE_ID'
        custom_order = ['Wide outside Off stump', 'Outside Off stump', 'Off stump', 'Middle stump', 'Leg stump', 'Outside Leg stump']

    fig = go.Figure()
    for bowler in match_shot_data['BowlerName'].unique():
        bowler_data = match_shot_data[(match_shot_data['BowlerName'] == bowler)]
        
        # Create the histogram trace for each bowler
        fig.add_trace(go.Histogram(
            x=[y for y in bowler_data[req_str] if y!=None],
            histnorm='probability',
            name=f'{bowler}', # name used in legend and hover labels
            opacity=0.75
        ))

    # Update the layout of the plot
    fig.update_layout(
        title_text=f'{type} Shot Data Probability Distribution', # title of plot
        xaxis_title_text=f'{type}', # xaxis label
        yaxis_title_text='Probability', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1, # gap between bars of the same location coordinates
        xaxis=dict(
            tickmode='array',
            tickvals=custom_order,  # Set the categories
            ticktext=custom_order,  # Set the text labels for those categories
            categoryorder='array',  # Specify that the order will be an array
            categoryarray=custom_order  # Provide the custom order of categories
        )        
    )

    st.plotly_chart(fig)

def speed_data(custom_df, plotno ,phase="All"):        

    if(len(custom_df) == 0):
        st.warning("Please Select a Bowling Type")
        st.empty()

    else:
        name_list = list(custom_df['BowlerName'].unique())
        # fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        # ax = ax.flatten()
        # fig1 = go.Figure()
        max_over_limit = custom_df['OverNo'].max()
        min_over_limit = custom_df['OverNo'].min()

        def input_pdf(input_list):
            input_list.sort()
            mean = np.mean(input_list)
            std = np.std(input_list)
            input_pdf_list = stats.norm.pdf(input_list, mean, std)
            return input_pdf_list
        
        def show_stump(ax):
            stump_height = 0.711
            stump_width = 0.2286 / 2
            stump_positions = [-stump_width, 0, stump_width]
            for pos in stump_positions:
                ax.plot([pos, pos], [0, stump_height], color='brown', linewidth=3)
            bail_y = stump_height  # Bails are at the top of the stumps
            ax.plot([stump_positions[0], stump_positions[1]], [bail_y, bail_y], color='brown', linewidth=2)  # Off-stump to Middle-stump
            ax.plot([stump_positions[1], stump_positions[2]], [bail_y, bail_y], color='brown', linewidth=2)  # Middle-stump to Leg-stump

        def show_pitch_top(ax):
            stump_width = 0.2286 / 2
            stump_positions = [-stump_width, 0, stump_width]

            for pos in stump_positions:
                ax.plot([pos, pos], [-2, 14], color='brown', linewidth=1)
            ax.plot([-1.83, 1.83], [1.22, 1.22], color='blue', linewidth=2)
            ax.plot([-1.32, -1.32], [-2, 1.22], color='blue', linewidth=2)
            ax.plot([1.32, 1.32], [-2, 1.22], color='blue', linewidth=2)

            ax.plot([-1.83, -1.83], [14, -2], color='brown', linewidth=2)
            ax.plot([1.83, 1.83], [14, -2], color='brown', linewidth=2)

            ax.plot([-stump_width, 0, stump_width], [0, 0, 0], 'o', color='brown', ms=2)
            ax.xlim(-1.83, 1.83)
            # ax.ylim(-2, 14)
            ax.ylim(14, -2)
            ax.xlim(-4, 4)

        # def gaussian_fun(input_list):
        #     kde = stats.gaussian_kde(input_list)
        #     gauss_x = np.linspace(min(input_list), max(input_list), 100)
        #     gauss_y = kde(gauss_x)
            # gauss_y_norm = gauss_y / np.trapz(gauss_y, gauss_x) 
            # return gauss_x, gauss_y_norm
        if(plotno < 10):

            for name in name_list:
                df = custom_df
                df_bowler = df[df['BowlerName'].str.contains(f"{name}", case=False, na=False)]
                bowler_name = df_bowler['BowlerName'].unique()[0]
                # df_bowler = df_bowler[(df_bowler['OverNo'] > 10) & (df_bowler['OverNo'] <= 40)]

                if(len(df_bowler) > 0):

                    if(plotno == 1):
                        speed_list = [float("%.3f"%(x*1.60934)) for x in df_bowler['release_speed']]
                        speed_list = [k for k in speed_list if k<165]
                        # speed_pdf = input_pdf(speed_list)
                        speed_mean = np.mean(speed_list)
                        sns.kdeplot(speed_list, label=f"{bowler_name} {speed_mean:.2f}")
                        bin_edges = np.arange(min(speed_list), max(speed_list), 0.5)
                        # sns.histplot(speed_list, bins=bin_edges, stat="probability", element='step', alpha=0.1)
                        # ax[0].plot(speed_list, speed_pdf, label = f"{bowler_name}")
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Speed Probability Distibution")
                        plt.legend()
                        plt.grid(True, linestyle='--')
                        
                    elif(plotno == 2):
                        length_list = [float("%.2f"%x) for x in df_bowler['bounce_x'] if x>=0]
                        # length_pdf = input_pdf(length_list)
                        avg_length = np.mean(length_list)
                        sns.kdeplot(length_list, label=f"{bowler_name} {avg_length:.2f}")
                        plt.xlim(0, 16)
                        bin_edges = np.arange(0, 16, 0.5)
                        # sns.histplot(length_list, bins=bin_edges, stat="probability", element="step", alpha = 0.5)
                        # bin_edges = np.arange(0, 16, 0.5)
                        # sns.histplot(length_list, bins=bin_edges, label=f"{bowler_name} {avg_length:.2f}", stat="probability")
                        # ax[1].plot(length_list,length_pdf, label=f"{bowler_name}")
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Length Probability Distribution")
                        plt.legend()
                        plt.grid(True, linestyle='--')

                    elif(plotno == 3):
                        line_list = [float("%.2f"%x) for x in df_bowler['stump_y'] if -5<x<5]
                        # line_pdf = input_pdf(line_list)
                        avg_line = np.mean([abs(x) for x in line_list])
                        sns.kdeplot(line_list, label=f"{bowler_name} {avg_line:.2f}")
                        # sns.histplot(line_list, bins=10, stat="probability", ax=ax[2], label=f"{bowler_name}")
                        plt.xlim(-1.75, 1.75)
                        bin_edges = np.arange(-2, 2, 0.1)
                        # sns.histplot(line_list, bins=bin_edges, stat="probability", element="step", alpha = 0.5)
                        # gauss_x, gauss_y = gaussian_fun(line_list)
                        # ax[2].plot(gauss_x,gauss_y, label=f"{bowler_name}")
                        # ax[2].plot(line_list,line_pdf, label=f"{bowler_name}")
                        show_stump(plt)
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Line Probability Distribution")
                        plt.legend()
                        plt.grid(True, linestyle='--')

                    elif(plotno == 4):
                        swing_list = [float("%.2f"%x) for x in df_bowler['swing'] if -50<x<50]
                        # swing_pdf = input_pdf(swing_list)
                        avg_swing = np.mean([abs(x) for x in swing_list])
                        sns.kdeplot(swing_list, label=f"{bowler_name} {avg_swing:.2f}")
                        bin_edges = np.arange(-8, 8, 0.5)
                        # sns.histplot(swing_list, bins=bin_edges, stat="probability", element="step", alpha = 0.5)
                        # ax[3].plot(swing_list,swing_pdf, label=f"{bowler_name}")
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Swing Probability Distribution")
                        plt.legend()
                        plt.grid(True, linestyle='--')

                    elif(plotno == 5):
                        seam_list = [float("%.2f"%x) for x in df_bowler['deviation'] if -50<x<50]
                        # seam_pdf = input_pdf(seam_list)
                        avg_seam = np.mean([abs(x) for x in seam_list])
                        sns.kdeplot(seam_list, label=f"{bowler_name} {avg_seam:.2f}")
                        bin_edges = np.arange(-10, 10, 0.5)
                        # sns.histplot(seam_list, bins=bin_edges, stat="probability", element="step", alpha = 0.5)
                        # ax[4].plot(seam_list,seam_pdf, label=f"{bowler_name}")
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Seam Probability Distribution")
                        plt.legend()
                        plt.grid(True, linestyle='--')

                    elif(plotno == 6):
                        release_y_list = [float("%.2f"%y) for y, z in zip(df_bowler['release_y'], df_bowler['release_z']) if -50<y<50 and -50<z<50]
                        release_z_list = [float("%.2f"%z) for y, z in zip(df_bowler['release_y'], df_bowler['release_z']) if -50<y<50 and -50<z<50]
                        show_stump(plt)
                        plt.plot(release_y_list,release_z_list, 'o',label=f"{bowler_name}")
                        plt.xlim(-1.75, 1.75)
                        plt.ylim(0, 2.5)
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Release Points Distribution")
                        plt.legend()
                        plt.grid(True, linestyle='--')
                    
                    elif(plotno == 7):
                        release_y_list = [float("%.2f"%y) for y, z in zip(df_bowler['stump_y'], df_bowler['stump_z']) if -50<y<50 and -50<z<50]
                        release_z_list = [float("%.2f"%z) for y, z in zip(df_bowler['stump_y'], df_bowler['stump_z']) if -50<y<50 and -50<z<50]
                        balls_hitting = [(y, z) for y,z in zip(release_y_list, release_z_list) if (z<0.711)&(y<0.1143)&(y>(-0.1143))]
                        perc_hitting = (len(balls_hitting)*100)/len(release_y_list)
                        show_stump(plt)
                        plt.plot(release_y_list,release_z_list, 'o',label=f"{bowler_name} {perc_hitting:.2f}%")
                        plt.xlim(-1.75, 1.75)
                        plt.ylim(0, 2.5)
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Beehive Distribution")
                        plt.legend()
                        plt.grid(True, linestyle='--')

                    elif(plotno == 8):
                        # pitch_x_list = [float("%.2f"%x) for x in df_bowler['bounce_x'] if x>0]
                        # pitch_y_list = [float("%.2f"%x) for x in df_bowler['bounce_y'] if -50<x<50]
                        plot_df = df_bowler
                        plot_df['bounce_y'] = plot_df['bounce_y']
                        pitch_x_list = [float("%.2f"%x) for x, y in zip(plot_df['bounce_x'], plot_df['bounce_y']) if x>-100 and -500<y<500]
                        pitch_y_list = [float("%.2f"%y) for x, y in zip(plot_df['bounce_x'], plot_df['bounce_y']) if x>-100 and -500<y<500]
                        # show_stump(plt)
                        plt.plot(pitch_y_list,pitch_x_list, 'o',label=f"{bowler_name}", markersize=5)
                        show_pitch_top(plt)
                        # plt.gca()
                        plt.title(f"Overs {min_over_limit}-{max_over_limit} Pitch Distribution")
                        plt.grid(False)
                        plt.legend()
            
            if(plotno == 9):
                beehive_df = custom_df[custom_df['stump_y'].between(-50, 50) & custom_df['stump_z'].between(-50, 50)].copy()
                boundaries_df = beehive_df[(beehive_df['IsFour'] == 1) | (beehive_df['IsSix'] == 1)]
                wickets_df = beehive_df[(beehive_df['IsWicket'] == 1)]
                dots_df = beehive_df[(beehive_df['IsDotball'] == 1)]
                runs_df = beehive_df[pd.to_numeric(beehive_df['BallRuns'], errors='coerce').fillna(0).astype(int).gt(0) & (beehive_df['IsFour'] == 0) & (beehive_df['IsSix'] == 0)]
                show_stump(plt)
                plt.plot(dots_df['stump_y'],dots_df['stump_z'], 'o', color='green',label="Dots", alpha=0.5)
                plt.plot(runs_df['stump_y'],runs_df['stump_z'], 'o', color='yellow',label="Runs", alpha=0.5)
                plt.plot(boundaries_df['stump_y'],boundaries_df['stump_z'], 'o', color='red',label="4s/6s")
                plt.plot(wickets_df['stump_y'],wickets_df['stump_z'], 'o', color='blue',label="Wickets")
                plt.title(f"Overs {min_over_limit}-{max_over_limit} Beehive Distribution")
                plt.xlim(-1.75, 1.75)
                plt.ylim(0, 2.5)
                plt.legend()
                plt.grid(True, linestyle='--')
                
            
            # plt.show()
            st.pyplot(plt)

        if(plotno == 10):
            bowler_name = st.selectbox(
                'Bowler',
                custom_df['BowlerName'].unique() if not custom_df.empty else [],
                index=None,
                placeholder='Choose an bowler',
            )

            if bowler_name:

                if custom_df.empty:
                    st.error("No data available for plotting.")
                else:
                    new_df_bowler = custom_df[custom_df['BowlerName'] == bowler_name].copy()
                    xy_rh = np.array(new_df_bowler[new_df_bowler['bounce_x'] > 0].assign(bounce_y=-new_df_bowler['bounce_y'])[['bounce_y', 'bounce_x']])
                    # new_df_bowler['bounce_y'] = -new_df_bowler['bounce_y']
                    # balls = new_df_bowler[new_df_bowler['bounce_x'] > 0].copy()
                    # xy_rh = np.array(balls[['bounce_y', 'bounce_x']])
                    title = bowler_name
                    subtitle_1 = f'Pitch Heatmap | {series_name}: {match_name}'
                    subtitle_2 = f'Tracking enabled for {len(xy_rh)} balls between Overs {min_over_limit}-{max_over_limit}.'
                    fig = pitch_densitymap(xy_rh, title, subtitle_1, subtitle_2)
                    # plt.show()
                    st.pyplot(fig)
                
    # fig.update_layout(
    #     title="PowerPlay Speed Probability Distribution",
    #     xaxis_title="Speed (km/h)",
    #     yaxis_title="Density",
    #     legend_title="Bowler",
    #     template="plotly_white"
    # )
    # st.plotly_chart(fig1)
# plt.plot()

if(format_type and series_name and match_name):
    match_df = format_df[(format_df['CompetitionName'] == series_name) & (format_df['MatchOrder'] == match_name)]
    match_id = match_df['MatchID'].unique()[0]
    max_overs = match_df['MATCH_NO_OF_OVERS'].unique()[0]
    # max_shot_data_inns = shot_data_df[shot_data_df['MatchID'] == match_id]['InningsNo'].max()
    # max_shot_data_overs = shot_data_df[(shot_data_df['MatchID'] == match_id) & (shot_data_df['InningsNo'] == max_shot_data_inns)]['OverNo'].max()
    match_shot_data = shot_data_df[shot_data_df['MatchID'] == match_id]
    max_len_shot_data = len(match_shot_data)

    over_range = st.slider(
        "Overs", 0, max_overs, (0, max_overs)
    )

    innings_list = match_shot_data['InningsNo'].unique()
    innings_type = st.multiselect(
            'Innings',
            innings_list,
            default=innings_list,
            placeholder='Choose an option'
    )
    match_shot_data = match_shot_data[match_shot_data['InningsNo'].isin(innings_type)]

    try:
        hawk_eye_df = pd.read_csv(f"./bcci_hawkeye_data/{match_id}.csv", low_memory=False)
        max_len_hawk_eye_data = len(hawk_eye_df)
        # max_hawkeye_inns = hawk_eye_df['InningsNo'].max()
        # max_hawkeye_overs = hawk_eye_df['OverNo'].max()
        hawk_eye_df = hawk_eye_df[hawk_eye_df['OverNo'].between(over_range[0], over_range[1])]
        hawk_eye_delivery_type_list = hawk_eye_df['delivery_type'].dropna().unique()

        bowling_type = st.multiselect(
            'Bowling Type',
            hawk_eye_delivery_type_list,
            default=hawk_eye_delivery_type_list,
            placeholder='Choose an option'
        )
        
        hawkeye_bowling_df = hawk_eye_df[hawk_eye_df['delivery_type'].isin(bowling_type) & hawk_eye_df['InningsNo'].isin(innings_type)]

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
        
    final_match_shot_data = match_shot_data[match_shot_data['OverNo'].between(over_range[0], over_range[1])]
    available_shot_data = (final_match_shot_data["BOWLING_LENGTH_ID"].notna().any() and final_match_shot_data["BOWLING_LINE_ID"].notna().any())
    if(available_shot_data == False):
        st.warning("Shot Data Not Available")
    
    # line_length_dist(final_match_shot_data, "Line")
    # line_length_dist(final_match_shot_data, "Length")
    # fig = px.histogram(match_shot_data, x='BOWLING_LENGTH_ID', color='BowlerName', histnorm='probability', category_orders=dict(BOWLING_LENGTH_ID = ['Full Toss', 'Yorker', 'Full Length', 'Good Length', 'Short of Good Length', 'Short Length']))
    # plt.figure(figsize=(8, 6))
    
    # st.pyplot(plt)
    
if (format_type and series_name and match_name and available_shot_data and len(final_match_shot_data)> 0):

    if(hawk_eye_df is not None):
        selected_option = st.selectbox('Choose an option', ['Line Shot Data Probability Distribution', 'Length Shot Data Probability Distribution', 
                                                            'Speed Probability Distibution','Length Kernel Density Estimation',
                                                            'Line Kernel Density Estimation', 'Swing Probability Distribution',
                                                            'Seam Probability Distribution', 'Release Points Distribution',
                                                            'Beehive Distribution Bowler Comparison', 'Beehive Distribution Outcome Comparision','Pitch Distribution', 'Pitch Heatmap'])
    else:
        selected_option = st.selectbox('Choose an option', ['Line Shot Data Probability Distribution', 'Length Shot Data Probability Distribution'])

    def plotting_func(selected_option):
        if(selected_option == 'Line Shot Data Probability Distribution'):
            line_length_dist(final_match_shot_data, "Line")
        elif(selected_option == 'Length Shot Data Probability Distribution'):
            line_length_dist(final_match_shot_data, "Length")
        elif(selected_option == 'Speed Probability Distibution'):
            speed_data(hawkeye_bowling_df, 1)
        elif(selected_option == 'Length Kernel Density Estimation'):
            speed_data(hawkeye_bowling_df, 2)
        elif(selected_option == 'Line Kernel Density Estimation'):
            speed_data(hawkeye_bowling_df, 3)
        elif(selected_option == 'Swing Probability Distribution'):
            speed_data(hawkeye_bowling_df, 4)
        elif(selected_option == 'Seam Probability Distribution'):
            speed_data(hawkeye_bowling_df, 5)
        elif(selected_option == 'Release Points Distribution'):
            speed_data(hawkeye_bowling_df, 6)
        elif(selected_option == 'Beehive Distribution Bowler Comparison'):
            speed_data(hawkeye_bowling_df, 7)
        elif(selected_option == 'Pitch Distribution'):
            speed_data(hawkeye_bowling_df, 8)
        elif(selected_option == 'Beehive Distribution Outcome Comparision'):
            speed_data(hawkeye_bowling_df, 9)
        elif(selected_option == 'Pitch Heatmap'):
            speed_data(hawkeye_bowling_df, 10)

    if(selected_option):
        plotting_func(selected_option)
    # st.dataframe(match_shot_data, use_container_width=True)
